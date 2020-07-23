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

import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.LongBitSet;
import org.apache.lucene.util.packed.PackedInts;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.lease.Releasable;
import org.elasticsearch.common.lease.Releasables;
import org.elasticsearch.common.util.BigArrays;
import org.elasticsearch.common.util.ByteArray;
import org.elasticsearch.common.util.ByteUtils;
import org.elasticsearch.common.util.IntArray;
import org.elasticsearch.search.aggregations.metrics.AbstractHyperLogLog;
import org.elasticsearch.search.aggregations.metrics.AbstractLinearCounting;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Hyperloglog++ counter, implemented based on pseudo code from
 * http://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/40671.pdf and its appendix
 * https://docs.google.com/document/d/1gyjfMHy43U9OWBXxfaeG-3MjGzejW1dlpyMwEYAAWEI/view?fullscreen
 *
 * This implementation is different from the original implementation in that it uses a hash table instead of a sorted list for linear
 * counting. Although this requires more space and makes hyperloglog (which is less accurate) used sooner, this is also considerably faster.
 *
 * Trying to understand what this class does without having read the paper is considered adventurous.
 *
 * The HyperLogLogPlusPlus contains two algorithms, one for linear counting and the HyperLogLog algorithm. Initially hashes added to the
 * data structure are processed using the linear counting until a threshold defined by the precision is reached where the data is replayed
 * to the HyperLogLog algorithm and then this is used.
 *
 * It supports storing several HyperLogLogPlusPlus structures which are identified by a bucket number.
 */
public final class MultiHyperLogLog implements Releasable {

    private static final long[] HLLPRECISIONTOTHRESHOLDS = new long[] {
        2,
        5,
        11,
        23,
        47,
        95,
        191,
        383,
        767,
        1535,
        3071,
        6143,
        12287,
        24575,
        350000 };

    /**
     * Compute the required threshold for the given precision.
     */
    public static long thresholdFromPrecision(int precision) {
        if (precision < AbstractHyperLogLog.MIN_PRECISION) {
            throw new IllegalArgumentException("Min precision is " + AbstractHyperLogLog.MIN_PRECISION + ", got " + precision);
        }
        if (precision > AbstractHyperLogLog.MAX_PRECISION) {
            throw new IllegalArgumentException("Max precision is " + AbstractHyperLogLog.MAX_PRECISION + ", got " + precision);
        }
        return HLLPRECISIONTOTHRESHOLDS[precision - 4];
    }

    private final HyperLogLog hll;

    public MultiHyperLogLog(int precision, BigArrays bigArrays, long initialBucketCount) {
        hll = new HyperLogLog(bigArrays, initialBucketCount, precision);
    }

    public int precision() {
        return hll.precision();
    }

    public long maxBucket() {
        return hll.runLens.size() >>> hll.precision();
    }

    public void merge(long bucket, AbstractHyperLogLog other) {
        hll.ensureCapacity(bucket + 1);
        if (precision() != other.precision()) {
            throw new IllegalArgumentException();
        }
        hll.bucket = bucket;
        hll.merge(other);
    }

    public void addRunLen(long bucket, int register, int runLen) {
        hll.ensureCapacity(bucket + 1);
        hll.bucket = bucket;
        hll.addRunLen(register, runLen);
    }

    public long cardinality(long bucket) {
        hll.bucket = bucket;
        return hll.cardinality();
    }

    public AbstractHyperLogLog getHyperLogLog(long bucket) {
        hll.bucket = bucket;
        return hll;
    }

    @Override
    public void close() {
        Releasables.close(hll);
    }

    private Object getComparableData(long bucket) {
        hll.bucket = bucket;
        return hll.getComparableData();
    }

    public int hashCode(long bucket) {
        return Objects.hash(precision(), getComparableData(bucket));
    }

    public boolean equals(long bucket, MultiHyperLogLog other) {
        return Objects.equals(precision(), other.precision())
            && Objects.equals(getComparableData(bucket), other.getComparableData(bucket));
    }

    private static class HyperLogLog extends AbstractHyperLogLog implements Releasable {
        private final BigArrays bigArrays;
        private final HyperLogLogIterator iterator;
        // array for holding the runlens.
        private ByteArray runLens;
        // Defines the position of the data structure. Callers of this object should set this value
        // before calling any of the methods.
        protected long bucket;

        HyperLogLog(BigArrays bigArrays, long initialBucketCount, int precision) {
            super(precision);
            this.runLens =  bigArrays.newByteArray(initialBucketCount << precision);
            this.bigArrays = bigArrays;
            this.iterator = new HyperLogLogIterator(this, precision, m);
        }

        @Override
        protected void addRunLen(int register, int encoded) {
            final long bucketIndex = (bucket << p) + register;
            runLens.set(bucketIndex, (byte) Math.max(encoded, runLens.get(bucketIndex)));
        }

        @Override
        protected RunLenIterator getRunLens() {
            iterator.reset(bucket);
            return iterator;
        }

        protected Object getComparableData() {
            Map<Byte, Integer> values = new HashMap<>();
            for (long i = 0; i < runLens.size(); i++) {
                byte runLength = runLens.get((bucket << p) + i);
                Integer numOccurances = values.get(runLength);
                if (numOccurances == null) {
                    values.put(runLength, 1);
                } else {
                    values.put(runLength, numOccurances + 1);
                }
            }
            return values;
        }

        protected void ensureCapacity(long numBuckets) {
            runLens = bigArrays.grow(runLens, numBuckets << p);
        }

        @Override
        public void close() {
            Releasables.close(runLens);
        }
    }

    private static class HyperLogLogIterator implements AbstractHyperLogLog.RunLenIterator {

        private final HyperLogLog hll;
        private final int m, p;
        int pos;
        long start;
        private byte value;

        HyperLogLogIterator(HyperLogLog hll, int p, int m) {
            this.hll = hll;
            this.m = m;
            this.p = p;
        }

        void reset(long bucket) {
            pos = 0;
            start = bucket << p;
        }

        @Override
        public boolean next() {
            if (pos < m) {
                value = hll.runLens.get(start + pos);
                pos++;
                return true;
            }
            return false;
        }

        @Override
        public byte value() {
            return value;
        }
    }

}
