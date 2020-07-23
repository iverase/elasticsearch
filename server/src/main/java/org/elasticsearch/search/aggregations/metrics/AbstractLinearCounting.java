/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.elasticsearch.search.aggregations.metrics;

import org.apache.lucene.util.packed.PackedInts;

/**
 * Linear counter, implemented based on pseudo code from
 * http://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/40671.pdf and its appendix
 * https://docs.google.com/document/d/1gyjfMHy43U9OWBXxfaeG-3MjGzejW1dlpyMwEYAAWEI/view?fullscreen
 *
 * Trying to understand what this class does without having read the paper is considered adventurous.
 *
 * The algorithm just keep a record of all distinct values provided encoded as an integer.
 */
public abstract class AbstractLinearCounting extends AbstractCardinalityAlgorithm {

    private static final int P2 = 25;

    public AbstractLinearCounting(int precision) {
       super(precision);
    }

    /**
     * Add encoded value to the linear counting. Implementor should only accept the value if it has not been
     * seen before.
     */
    protected abstract int addEncoded(int encoded);

    /**
     * number of values in the counter.
     */
    protected abstract int size();

    /**
     * return the current values in the counter.
     */
    protected abstract HashesIterator values();

    public int collect(long hash) {
        final int k = encodeHash(hash, p);
        return addEncoded(k);
    }

    public long cardinality() {
        final long m = 1 << P2;
        final long v = m - size();
        return linearCounting(m, v);
    }

    static long mask(int bits) {
        return (1L << bits) - 1;
    }

    /**
     * Encode the hash on 32 bits. The encoded hash cannot be equal to <code>0</code>.
     */
    static int encodeHash(long hash, int p) {
        final long e = hash >>> (64 - P2);
        final long encoded;
        if ((e & mask(P2 - p)) == 0) {
            final int runLen = 1 + Math.min(Long.numberOfLeadingZeros(hash << P2), 64 - P2);
            encoded = (e << 7) | (runLen << 1) | 1;
        } else {
            encoded = e << 1;
        }
        assert PackedInts.bitsRequired(encoded) <= 32;
        assert encoded != 0;
        return (int) encoded;
    }

    /** Iterator over the hash values */
    public interface HashesIterator {

        /**
         * number of elements in the iterator
         */
        long size();
        /**
         * Moves the iterator to the next element if it exists.
         * @return true if there is a next value, else false.
         */
        boolean next();

        /**
         * Hash value.
         * @return the current value of the counter.
         */
        int value();
    }
}
