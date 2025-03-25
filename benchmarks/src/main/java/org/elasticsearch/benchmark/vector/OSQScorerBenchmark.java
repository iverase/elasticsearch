/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.benchmark.vector;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.simdvec.VectorScorerFactory;
import org.elasticsearch.simdvec.internal.vectorization.OSQVectorsScorer;
import org.elasticsearch.simdvec.internal.vectorization.OptimizedScalarQuantizer;
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
import java.nio.file.Files;
import java.util.Random;
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
 * Run with ./gradlew -p benchmarks run --args 'VectorScorerBenchmark'
 */
public class OSQScorerBenchmark {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    @Param({"1024" })
    int dims;

    int length;

    int numVectors = OSQVectorsScorer.BULK_SIZE * 10;
    int numQueries = 10;

    byte[][] binaryVectors;
    byte[][] binaryQueries;
    OptimizedScalarQuantizer.QuantizationResult result;
    float centroidDp;

    byte[] scratch;
    OSQVectorsScorer scorer;

    IndexInput in;

    float[] scratchScores;
    float[] corrections;


    @Setup
    public void setup() throws IOException {
        Random random = new Random(123);

        this.length = OptimizedScalarQuantizer.discretize(dims, 64) / 8;

        binaryVectors = new byte[numVectors][length];
        for (byte[] binaryVector : binaryVectors) {
            random.nextBytes(binaryVector);
        }

        Directory dir = new MMapDirectory(Files.createTempDirectory("vectorData"));
        IndexOutput out = dir.createOutput("vectors", IOContext.DEFAULT);
        byte[] correctionBytes = new byte[14 * OSQVectorsScorer.BULK_SIZE];
        for (int i =0; i < numVectors; i+=OSQVectorsScorer.BULK_SIZE) {
            for (int j = 0; j < OSQVectorsScorer.BULK_SIZE; j++) {
                out.writeBytes(binaryVectors[i + j], 0, binaryVectors[i + j].length);
            }
            random.nextBytes(correctionBytes);
            out.writeBytes(correctionBytes, 0, correctionBytes.length);
        }
        out.close();
        in = dir.openInput("vectors", IOContext.DEFAULT);

        binaryQueries = new byte[numVectors][4 * length];
        for (byte[] binaryVector : binaryVectors) {
            random.nextBytes(binaryVector);
        }
        result = new OptimizedScalarQuantizer.QuantizationResult(
            random.nextFloat(), random.nextFloat(), random.nextFloat(), Short.toUnsignedInt((short) random.nextInt())
        );
        centroidDp = random.nextFloat();

        scratch = new byte[length];
        scorer = VectorScorerFactory.instance().get().getOSQVectorsScorer(in, length);
        scratchScores = new float[16];
        corrections = new float[3];
    }

    @Benchmark
    public void scoreFromArray(Blackhole bh) throws IOException {
        for (int j = 0; j < numQueries; j++) {
            in.seek(0);
            for (int i = 0; i < numVectors; i++) {
                in.readBytes(scratch, 0, length);
                float qDist = ESVectorUtil.ipByteBinByte(binaryQueries[i], scratch);
                in.readFloats(corrections, 0, corrections.length);
                int addition = Short.toUnsignedInt(in.readShort());
                float score = scorer.quantizeScore(result, VectorSimilarityFunction.EUCLIDEAN, centroidDp, corrections[0],
                    corrections[1], addition, corrections[2], qDist);
                bh.consume(score);
            }
        }
    }

    @Benchmark
    public void scoreFromMemorySegmentOnlyVector(Blackhole bh) throws IOException {
        for (int j = 0; j < numQueries; j++) {
            in.seek(0);
            for (int i = 0; i < numVectors; i++) {
                float qDist = scorer.int4BitDotProduct(binaryQueries[j]);
                in.readFloats(corrections, 0, corrections.length);
                int addition = Short.toUnsignedInt(in.readShort());
                float score = scorer.quantizeScore(result, VectorSimilarityFunction.EUCLIDEAN, centroidDp, corrections[0],
                    corrections[1], addition, corrections[2], qDist);
                bh.consume(score);
            }
        }
    }

    @Benchmark
    public void scoreFromMemorySegmentOnlyVectorBulk(Blackhole bh) throws IOException {
        for (int j = 0; j < numQueries; j++) {
            in.seek(0);
            for (int i = 0; i < numVectors; i += 16) {
                scorer.int4BitDotProductBulk(binaryQueries[j], OSQVectorsScorer.BULK_SIZE ,scratchScores);
                for (int k = 0; k < OSQVectorsScorer.BULK_SIZE; k++) {
                    in.readFloats(corrections, 0, corrections.length);
                    int addition = Short.toUnsignedInt(in.readShort());
                    float score = scorer.quantizeScore(result, VectorSimilarityFunction.EUCLIDEAN, centroidDp, corrections[0],
                        corrections[1], addition, corrections[2], scratchScores[k]);
                    bh.consume(score);
                }

            }
        }
    }

    @Benchmark
    public void scoreFromMemorySegmentAllBulk(Blackhole bh) throws IOException {
        for (int j = 0; j < numQueries; j++) {
            in.seek(0);
            for (int i = 0; i < numVectors; i += 16) {
                scorer.quantizeScoreBulk(binaryQueries[j], result, VectorSimilarityFunction.EUCLIDEAN, centroidDp, scratchScores);
                bh.consume(scratchScores);
            }
        }
    }
}
