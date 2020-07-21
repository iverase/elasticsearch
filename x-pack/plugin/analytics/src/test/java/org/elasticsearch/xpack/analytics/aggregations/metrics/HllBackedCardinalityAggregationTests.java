/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
package org.elasticsearch.xpack.analytics.aggregations.metrics;

import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.TestUtil;
import org.elasticsearch.action.admin.indices.mapping.put.PutMappingRequest;
import org.elasticsearch.action.admin.indices.refresh.RefreshRequest;
import org.elasticsearch.action.bulk.BulkRequest;
import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.common.util.BigArrays;
import org.elasticsearch.common.util.ByteArray;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.aggregations.AggregationBuilders;
import org.elasticsearch.search.aggregations.metrics.CardinalityAggregationBuilder;
import org.elasticsearch.search.aggregations.metrics.HyperLogLogPlusPlus;
import org.elasticsearch.search.aggregations.metrics.InternalCardinality;
import org.elasticsearch.test.ESSingleNodeTestCase;
import org.elasticsearch.xpack.analytics.AnalyticsPlugin;
import org.elasticsearch.xpack.core.LocalStateCompositeXPackPlugin;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


public class HllBackedCardinalityAggregationTests extends ESSingleNodeTestCase {

    public void testCardinalityAggregation() throws Exception {

        int precision = TestUtil.nextInt(random(), 4, 18);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
              .startObject("_doc")
                .startObject("properties")
                  .startObject("data")
                     .field("type", "keyword")
                  .endObject()
                .endObject()
              .endObject()
            .endObject();
        createIndex("raw");
        PutMappingRequest request = new PutMappingRequest("raw").source(xContentBuilder);
        client().admin().indices().putMapping(request).actionGet();


        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
              .startObject("_doc")
                .startObject("properties")
                  .startObject("data")
                    .field("type", "hll")
                    .field("precision", precision)
                  .endObject()
                .endObject()
              .endObject()
            .endObject();
        createIndex("pre_agg");
        PutMappingRequest request2 = new PutMappingRequest("pre_agg").source(xContentBuilder2);
        client().admin().indices().putMapping(request2).actionGet();

        HyperLogLogPlusPlus histogram = new HyperLogLogPlusPlus(precision, BigArrays.NON_RECYCLING_INSTANCE, 1);
        final org.elasticsearch.common.hash.MurmurHash3.Hash128 hash = new org.elasticsearch.common.hash.MurmurHash3.Hash128();
        BulkRequest bulkRequest = new BulkRequest();

        int numDocs = 100000;
        int frq = 10000;

        ByteArray byteArray =  BigArrays.NON_RECYCLING_INSTANCE.newByteArray(1 << precision);
        int[] tmp = new int[1 << precision];

        for (int i =0; i < numDocs; i ++) {
            String value  = TestUtil.randomSimpleString(random());
            XContentBuilder doc = XContentFactory.jsonBuilder()
                .startObject()
                  .field("data", value)
                .endObject();
            bulkRequest.add(new IndexRequest("raw").source(doc));
            BytesRef ref = new BytesRef(value);
            org.elasticsearch.common.hash.MurmurHash3.hash128(ref.bytes, ref.offset, ref.length, 0, hash);
            histogram.collect(0, hash.h1);
            if ((i + 1) % frq == 0) {
                client().bulk(bulkRequest);
                bulkRequest = new BulkRequest();
                histogram.getRunLens(0, byteArray);
                XContentBuilder preAggDoc = XContentFactory.jsonBuilder()
                    .startObject()
                      .startObject("data")
                        .field("sketch", toArray(byteArray, tmp))
                      .endObject()
                    .endObject();
                client().prepareIndex("pre_agg").setSource(preAggDoc).get();
                histogram =  new HyperLogLogPlusPlus(precision, BigArrays.NON_RECYCLING_INSTANCE, 1);
            }
        }
        client().admin().indices().refresh(new RefreshRequest("raw", "pre_agg")).get();

        SearchResponse response = client().prepareSearch("raw").setTrackTotalHits(true).get();
        assertEquals(numDocs, response.getHits().getTotalHits().value);

        response = client().prepareSearch("pre_agg").get();
        assertEquals(numDocs / frq, response.getHits().getTotalHits().value);

        int aggPrecision = TestUtil.nextInt(random(), 4, precision);

        CardinalityAggregationBuilder builder =
            AggregationBuilders.cardinality("agg").field("data")
                .precisionThreshold(HyperLogLogPlusPlus.thresholdFromPrecision(aggPrecision));

        SearchResponse responseRaw = client().prepareSearch("raw").addAggregation(builder).get();
        SearchResponse responsePreAgg = client().prepareSearch("pre_agg").addAggregation(builder).get();
        SearchResponse responseBoth = client().prepareSearch("pre_agg", "raw").addAggregation(builder).get();

        InternalCardinality cardinalityRaw =  responseRaw.getAggregations().get("agg");
        InternalCardinality cardinalityPreAgg =  responsePreAgg.getAggregations().get("agg");
        assertEquals(cardinalityRaw.getValue(), cardinalityPreAgg.getValue());
        InternalCardinality cardinalityBoth =  responseBoth.getAggregations().get("agg");
        assertEquals(cardinalityBoth.getValue(), cardinalityPreAgg.getValue());
    }

    public void testCardinalityAggregationWithPath() throws Exception {

        int precision = TestUtil.nextInt(random(), 4, 18);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
              .startObject("_doc")
                .startObject("properties")
                    .startObject("parent")
                      .startObject("properties")
                        .startObject("data")
                          .field("type", "keyword")
                        .endObject()
                      .endObject()
                    .endObject()
                 .endObject()
               .endObject()
            .endObject();
        createIndex("raw");
        PutMappingRequest request = new PutMappingRequest("raw").source(xContentBuilder);
        client().admin().indices().putMapping(request).actionGet();


        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
              .startObject("_doc")
                .startObject("properties")
                  .startObject("parent")
                     .startObject("properties")
                       .startObject("data")
                         .field("type", "hll")
                         .field("precision", precision)
                       .endObject()
                     .endObject()
                   .endObject()
                 .endObject()
              .endObject()
            .endObject();
        createIndex("pre_agg");
        PutMappingRequest request2 = new PutMappingRequest("pre_agg").source(xContentBuilder2);
        client().admin().indices().putMapping(request2).actionGet();

        HyperLogLogPlusPlus histogram = new HyperLogLogPlusPlus(precision, BigArrays.NON_RECYCLING_INSTANCE, 1);
        final org.elasticsearch.common.hash.MurmurHash3.Hash128 hash = new org.elasticsearch.common.hash.MurmurHash3.Hash128();
        BulkRequest bulkRequest = new BulkRequest();

        int numDocs = 100000;
        int frq = 10000;

        ByteArray byteArray =  BigArrays.NON_RECYCLING_INSTANCE.newByteArray(1 << precision);
        int[] tmp = new int[1 << precision];

        for (int i =0; i < numDocs; i ++) {
            String value  = TestUtil.randomSimpleString(random());
            XContentBuilder doc = XContentFactory.jsonBuilder()
                .startObject()
                  .startObject("parent")
                    .field("data", value)
                   .endObject()
                .endObject();
            bulkRequest.add(new IndexRequest("raw").source(doc));
            BytesRef ref = new BytesRef(value);
            org.elasticsearch.common.hash.MurmurHash3.hash128(ref.bytes, ref.offset, ref.length, 0, hash);
            histogram.collect(0, hash.h1);
            if ((i + 1) % frq == 0) {
                client().bulk(bulkRequest);
                bulkRequest = new BulkRequest();
                histogram.getRunLens(0, byteArray);
                XContentBuilder preAggDoc = XContentFactory.jsonBuilder()
                    .startObject()
                      .startObject("parent")
                        .startObject("data")
                          .field("sketch", toArray(byteArray, tmp))
                        .endObject()
                      .endObject()
                    .endObject();
                client().prepareIndex("pre_agg").setSource(preAggDoc).get();
                histogram =  new HyperLogLogPlusPlus(precision, BigArrays.NON_RECYCLING_INSTANCE, 1);
            }
        }
        client().admin().indices().refresh(new RefreshRequest("raw", "pre_agg")).get();

        SearchResponse response = client().prepareSearch("raw").setTrackTotalHits(true).get();
        assertEquals(numDocs, response.getHits().getTotalHits().value);

        response = client().prepareSearch("pre_agg").get();
        assertEquals(numDocs / frq, response.getHits().getTotalHits().value);

        int aggPrecision = TestUtil.nextInt(random(), 4, precision);

        CardinalityAggregationBuilder builder =
            AggregationBuilders.cardinality("agg").field("parent.data")
                .precisionThreshold(HyperLogLogPlusPlus.thresholdFromPrecision(aggPrecision));

        SearchResponse responseRaw = client().prepareSearch("raw").addAggregation(builder).get();
        SearchResponse responsePreAgg = client().prepareSearch("pre_agg").addAggregation(builder).get();
        SearchResponse responseBoth = client().prepareSearch("pre_agg", "raw").addAggregation(builder).get();

        InternalCardinality cardinalityRaw =  responseRaw.getAggregations().get("agg");
        InternalCardinality cardinalityPreAgg =  responsePreAgg.getAggregations().get("agg");
        assertEquals(cardinalityRaw.getValue(), cardinalityPreAgg.getValue());
        InternalCardinality cardinalityBoth =  responseBoth.getAggregations().get("agg");
        assertEquals(cardinalityBoth.getValue(), cardinalityPreAgg.getValue());
    }

    private int[] toArray(ByteArray byteArray, int[] collector) {
        for (int i = 0; i < byteArray.size(); i++) {
            collector[i] = byteArray.get(i);
        }
        return collector;
    }

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        List<Class<? extends Plugin>> plugins = new ArrayList<>(super.getPlugins());
        plugins.add(AnalyticsPlugin.class);
        plugins.add(LocalStateCompositeXPackPlugin.class);
        return plugins;
    }

}
