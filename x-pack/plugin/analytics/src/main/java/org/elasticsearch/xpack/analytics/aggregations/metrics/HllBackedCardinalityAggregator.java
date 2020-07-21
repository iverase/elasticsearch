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
                                               LeafBucketCollector sub) throws IOException {
        if (valuesSource == null) {
            return LeafBucketCollector.NO_OP_COLLECTOR;
        }
        HllValuesSource.HllSketch source = (HllValuesSource.HllSketch) valuesSource;
        if (precision == fieldPrecision) {
            return new EqualPrecisionHllCollector(counts, source.getHllValues(ctx), collectorArray);
        } else {
            return new DifferentPrecisionHllCollector(counts, source.getHllValues(ctx), collectorArray, fieldPrecision);
        }
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
                    final byte runLen = mergeRegister(value);
                    tmp.set(i, runLen);
                }
                assert value.next() == false;
                counts.collectRunLens(bucketOrd, tmp);
            }
        }

        private byte mergeRegister(HllValue value) throws IOException {
            value.next();
            final byte runLen = value.value();
            if (runLen != 0) {
                // If the first element is set, then runLen is this value plus the change in precision
                value.skip(registersToMerge - 1);
                return (byte) (runLen + precisionDiff);
            } else {
                // Find the first set value and compute the runLen for the precision change
                for (int i = 1; i < registersToMerge; i++) {
                    value.next();
                    if (value.value() != 0) {
                        value.skip(registersToMerge - i - 1);
                        return (byte) (precisionDiff - (int) (Math.log(i) / Math.log(2)));
                    }
                }
            }
            return 0;
        }
    }
}
