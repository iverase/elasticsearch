/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
    private final int fieldPrecision;
    private final ValuesSource valuesSource;
    @Nullable
    private final HyperLogLogPlusPlus counts;
    @Nullable
    private final ByteArray collectorArray;

    private LeafBucketCollector collector;

    public HllBackedCardinalityAggregator(
            String name,
            ValuesSourceConfig valuesSourceConfig,
            int precision,
            int fieldPrecision,
            SearchContext context,
            Aggregator parent,
            Map<String, Object> metadata) throws IOException {
        super(name, context, parent, metadata);
        // TODO: Stop using nulls here
        this.valuesSource = valuesSourceConfig.hasValues() ? valuesSourceConfig.getValuesSource() : null;
        this.precision = precision;
        this.fieldPrecision = fieldPrecision;
        if (valuesSource == null) {
            this.counts = null;
            this.collectorArray = null;
        } else {
            this.counts = new HyperLogLogPlusPlus(precision, context.bigArrays(), 1);
            this.collectorArray = context.bigArrays().newByteArray(1 << precision);
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
        if (precision == fieldPrecision) {
            collector = new EqualPrecisionHllCollector(counts, source.getHllValues(ctx), collectorArray);
        } else {
            collector = new DifferentPrecisionHllCollector(counts, source.getHllValues(ctx), collectorArray, fieldPrecision);
        }
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
        copy.merge(0, counts, owningBucketOrdinal);
        return new InternalCardinality(name, copy, metadata());
    }

    @Override
    public InternalAggregation buildEmptyAggregation() {
        return new InternalCardinality(name, null, metadata());
    }

    @Override
    protected void doClose() {
        Releasables.close(counts, collectorArray);
    }

    private static class EmptyCollector extends LeafBucketCollector {

        @Override
        public void collect(int doc, long bucketOrd) {
            // no-op
        }
    }

    private static class EqualPrecisionHllCollector extends LeafBucketCollector {

        private final HllValues values;
        private final HyperLogLogPlusPlus counts;
        private final ByteArray tmp;
        final int m;

        EqualPrecisionHllCollector(HyperLogLogPlusPlus counts, HllValues values, ByteArray byteArray) {
            this.counts = counts;
            this.values = values;
            this.tmp = byteArray;
            this.m = 1 << counts.precision();
        }

        @Override
        public void collect(int doc, long bucketOrd) throws IOException {
            if (values.advanceExact(doc)) {
                final HllValue value = values.hllValue();
                for (int i = 0; i < m; i++) {
                    value.next();
                    tmp.set(i, value.value());
                }
                assert value.next() == false;
                counts.collectRunLens(bucketOrd, tmp);
            }
        }
    }

    private static class DifferentPrecisionHllCollector extends LeafBucketCollector {

        private final HllValues values;
        private final HyperLogLogPlusPlus counts;
        private final ByteArray tmp;
        final int m;
        final int precisionDiff;
        final int registersToMerge;

        DifferentPrecisionHllCollector(HyperLogLogPlusPlus counts,
                                       HllValues values,
                                       ByteArray byteArray,
                                       int fieldPrecision) {
            this.counts = counts;
            this.values = values;
            this.tmp = byteArray;
            this.m = 1 << counts.precision();
            this.precisionDiff = fieldPrecision - counts.precision();
            this.registersToMerge = 1 << precisionDiff;
        }

        @Override
        public void collect(int doc, long bucketOrd) throws IOException {
            if (values.advanceExact(doc)) {
                final HllValue value = values.hllValue();
                for (int i = 0; i < m; i++) {
                    value.next();
                    final byte runLen = value.value();
                    if (runLen != 0) {
                        // If the first element is set, then runLen is this value plus the change in precision
                        tmp.set(i, (byte) (runLen + precisionDiff));
                        value.skip(registersToMerge - 1);
                    } else {
                        tmp.set(i, (byte) 0);
                        // Find the first set value and compute the runLen for the precision change
                        for (int j = 1; j < registersToMerge; j++) {
                            value.next();
                            if (value.value() != 0) {
                                tmp.set(i, (byte) (precisionDiff - (int) (Math.log(j) / Math.log(2))));
                                value.skip(registersToMerge - j - 1);
                                break;
                            }
                        }
                    }
                }
                assert value.next() == false;
                counts.collectRunLens(bucketOrd, tmp);
            }
        }
    }
}
