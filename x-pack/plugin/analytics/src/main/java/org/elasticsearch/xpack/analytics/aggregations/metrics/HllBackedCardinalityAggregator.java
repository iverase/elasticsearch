/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.elasticsearch.xpack.analytics.aggregations.metrics;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.ScoreMode;
import org.elasticsearch.common.Nullable;
import org.elasticsearch.common.lease.Releasables;
import org.elasticsearch.common.util.BigArrays;
import org.elasticsearch.common.util.ByteArray;
import org.elasticsearch.search.aggregations.Aggregator;
import org.elasticsearch.search.aggregations.InternalAggregation;
import org.elasticsearch.search.aggregations.LeafBucketCollector;
import org.elasticsearch.search.aggregations.metrics.HyperLogLogPlusPlus;
import org.elasticsearch.search.aggregations.metrics.InternalCardinality;
import org.elasticsearch.search.aggregations.metrics.NumericMetricsAggregator;
import org.elasticsearch.search.aggregations.support.ValuesSource;
import org.elasticsearch.search.aggregations.support.ValuesSourceConfig;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.xpack.analytics.aggregations.support.HllValuesSource;
import org.elasticsearch.xpack.analytics.mapper.fielddata.HllValue;
import org.elasticsearch.xpack.analytics.mapper.fielddata.HllValues;

import java.io.IOException;
import java.util.Map;

/**
 * An aggregator that computes approximate counts of unique values from Hll sketches.
 */
public class HllBackedCardinalityAggregator extends NumericMetricsAggregator.SingleValue {

    private final int precision;
    private final ValuesSource valuesSource;
    @Nullable
    private final HyperLogLogPlusPlus counts;
    @Nullable
    private final ByteArray fieldArray;
    @Nullable
    private final ByteArray finalArray;

    private HllCollector collector;

    public HllBackedCardinalityAggregator(
            String name,
            ValuesSourceConfig valuesSourceConfig,
            int precision,
            int fieldTypePrecision,
            SearchContext context,
            Aggregator parent,
            Map<String, Object> metadata) throws IOException {
        super(name, context, parent, metadata);
        // TODO: Stop using nulls here
        this.valuesSource = valuesSourceConfig.hasValues() ? valuesSourceConfig.getValuesSource() : null;
        this.precision = precision;
        if (valuesSource == null) {
            this.counts = null;
            this.fieldArray = null;
            this.finalArray = null;
        } else {
            this.counts = new HyperLogLogPlusPlus(fieldTypePrecision, context.bigArrays(), 1);
            this.fieldArray = context.bigArrays().newByteArray(1 << fieldTypePrecision);
            // Only used if we need to decrease the precision of the HLL sketch
            this.finalArray = precision == fieldTypePrecision ? null :  context.bigArrays().newByteArray(1 << precision);
        }
    }

    @Override
    public ScoreMode scoreMode() {
        return valuesSource != null && valuesSource.needsScores() ? ScoreMode.COMPLETE : ScoreMode.COMPLETE_NO_SCORES;
    }

    @Override
    public LeafBucketCollector getLeafCollector(LeafReaderContext ctx,
            final LeafBucketCollector sub) throws IOException {
        if (valuesSource == null) {
            return new EmptyCollector();
        }
        HllValuesSource.HllSketch source = (HllValuesSource.HllSketch) valuesSource;
        collector = new HllCollector(counts, source.getHllValues(ctx), fieldArray);
        return collector;
    }

    @Override
    public double metric(long owningBucketOrd) {
        return counts == null ? 0 : counts.cardinality(owningBucketOrd);
    }

    @Override
    public InternalAggregation buildAggregation(long owningBucketOrdinal) {
        if (counts == null || owningBucketOrdinal >= counts.maxBucket() ||
            counts.cardinality(owningBucketOrdinal) == 0) {
            return buildEmptyAggregation();
        }
        // We need to build a copy because the returned Aggregation needs remain usable after
        // this Aggregator (and its HLL++ counters) is released.
        HyperLogLogPlusPlus copy = new HyperLogLogPlusPlus(precision, BigArrays.NON_RECYCLING_INSTANCE, 1);
        if (precision == counts.precision()) {
            copy.merge(0, counts, owningBucketOrdinal);
        } else {
            // We need to reduce the precision of the sketch
            counts.getRunLens(owningBucketOrdinal, fieldArray);
            reducePrecision(counts.precision(), precision, fieldArray, finalArray);
            copy.collectRunLens(0, finalArray);
        }
        return new InternalCardinality(name, copy, metadata());
    }

    private static void reducePrecision(int initialPrecision,
                                        int finalPrecision,
                                        ByteArray initialArray,
                                        ByteArray finalArray) {
        assert initialPrecision >= finalPrecision;
        assert initialArray.size() == 1 << initialPrecision;
        assert finalArray.size() == 1 << finalPrecision;

        final int m = 1 << initialPrecision;
        final int precisionDiff = initialPrecision - finalPrecision;
        final int registersToMerge = 1 << precisionDiff;

        for (int i = 0, j = 0; i < m; i += registersToMerge, j++) {
            final byte runLen = initialArray.get(i);
            if (runLen != 0) {
                // If the first element is set, then runLen is this value plus the change in precision
                finalArray.set(j, (byte) (runLen + precisionDiff));
            } else {
                // Find the first set value and compute the runLen for the precision change
                for (int k = 1; k < registersToMerge; k++) {
                    if (initialArray.get(i + k) != 0) {
                        finalArray.set(j, (byte) (precisionDiff - (int) (Math.log(k) / Math.log(2))));
                        break;
                    }
                }
            }
        }
    }

    @Override
    public InternalAggregation buildEmptyAggregation() {
        return new InternalCardinality(name, null, metadata());
    }

    @Override
    protected void doClose() {
        if (collector != null) {
            Releasables.close(counts, fieldArray, finalArray);
        }
    }

    private static class EmptyCollector extends LeafBucketCollector {

        @Override
        public void collect(int doc, long bucketOrd) {
            // no-op
        }
    }

    private static class HllCollector extends LeafBucketCollector {

        private final HllValues values;
        private final HyperLogLogPlusPlus counts;
        private final ByteArray tmp;

        HllCollector(HyperLogLogPlusPlus counts, HllValues values, ByteArray byteArray) {
            this.counts = counts;
            this.values = values;
            this.tmp = byteArray;
        }

        @Override
        public void collect(int doc, long bucketOrd) throws IOException {
            if (values.advanceExact(doc)) {
                final HllValue value = values.hllValue();
                int i = 0;
                while(value.next()) {
                    tmp.set(i++, value.value());
                }
                assert i == 1 << counts.precision();
            }
            counts.collectRunLens(bucketOrd, tmp);
        }
    }
}
