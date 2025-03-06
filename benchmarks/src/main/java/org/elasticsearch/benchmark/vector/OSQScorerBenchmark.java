/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.vector;

import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.BQSpaceUtils;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;


@Fork(value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector" })
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
/**
 * Benchmark that compares various scalar quantized vector similarity function
 * implementations;: scalar, lucene's panama-ized, and Elasticsearch's native.
 * Run with ./gradlew -p benchmarks run --args 'OSQScorerBenchmark'
 */
public class OSQScorerBenchmark {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    @Param({"1024" })
    int dims;

    int size;

    int numVectors = 1600;
    int numQueries = 10;

    byte[][] packedVector;
    byte[][] packedQuery;

    byte[][] bulkVectors;


    @Setup
    public void setup() throws IOException {


        size = dims / 8;

        packedVector = new byte[numVectors][size];
        for (int i = 0; i < numVectors; i++) {
            ThreadLocalRandom.current().nextBytes(packedVector[i]);
        }
        packedQuery = new byte[numQueries][BQSpaceUtils.B_QUERY * size];
        for (int i = 0; i < numQueries; i++) {
            ThreadLocalRandom.current().nextBytes(packedQuery[i]);
        }
        bulkVectors = new byte[numVectors/ 16][16 * size];
        for (int i = 0; i < bulkVectors.length; i++) {
            for (int j = 0; j < 16; j++) {
                System.arraycopy(packedVector[16 * i + j], 0, bulkVectors[i], j * size, size);
            }
        }
    }


    @Benchmark
    public void scoreIpByteBinByte(Blackhole bh) {
        for (byte[] query :  packedQuery) {
            for (byte[] vector : packedVector) {
                bh.consume(ESVectorUtil.ipByteBinByte(query, vector));
            }
        }
    }

    @Benchmark
    public void scoreIpByteBinByteBulk(Blackhole bh) {
        long[] output = new long[16];
        for (byte[] query :  packedQuery) {
            for (byte[] vector : bulkVectors) {
                Arrays.fill(output,  0L);
                ESVectorUtil.ipByteBinByteBulk(query, vector, size, 16, output);
                bh.consume(output);
            }
        }
    }
}
