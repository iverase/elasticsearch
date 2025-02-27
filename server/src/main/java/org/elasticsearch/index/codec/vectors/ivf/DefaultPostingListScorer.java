/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ivf;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.BQVectorUtils;
import org.elasticsearch.index.codec.vectors.es818.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

public class DefaultPostingListScorer {

    private static final float FOUR_BIT_SCALE = 1f / ((1 << 4) - 1);

    public static void writePostingList(
        DataOutput output,
        int numberDimensions,
        OptimizedScalarQuantizer scalarQuantizer,
        int[] docIds,
        float[] centroid,
        float[][] vectors
    ) throws IOException {
        output.writeVInt(vectors.length);
        int discreteDims = BQVectorUtils.discretize(numberDimensions, 64);
        byte[] quantizationScratch = new byte[discreteDims];
        byte[] vector = new byte[discreteDims / 8];
        for (int i = 0; i < vectors.length; i++) {
            output.writeInt(docIds[i]);
            float[] v = vectors[i];
            OptimizedScalarQuantizer.QuantizationResult corrections = scalarQuantizer.scalarQuantize(
                v,
                quantizationScratch,
                (byte) 1,
                centroid
            );
            BQVectorUtils.packAsBinary(quantizationScratch, vector);
            output.writeBytes(vector, vector.length);
            output.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
            output.writeInt(Float.floatToIntBits(corrections.upperInterval() - corrections.lowerInterval()));
            output.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
            assert corrections.quantizedComponentSum() >= 0 && corrections.quantizedComponentSum() <= 0xffff;
            output.writeShort((short) corrections.quantizedComponentSum());
        }
    }

    public static boolean scorePostingList(
        int numberDimensions,
        VectorSimilarityFunction similarityFunction,
        DataInput input,
        byte[] query,
        OptimizedScalarQuantizer.QuantizationResult queryCorrections,
        KnnCollector knnCollector,
        Bits acceptDocs,
        float[] centroid
    ) throws IOException {
        byte[] binaryCode = new byte[BQVectorUtils.discretize(numberDimensions, 64) / 8];
        int vectors = input.readVInt();
        float centroidDp = VectorUtil.dotProduct(centroid, centroid);
        float[] correctiveValues = new float[3];
        for (int i = 0; i < vectors; i++) {
            final int docId = input.readInt();
            if (acceptDocs != null && acceptDocs.get(docId) == false) {
                input.skipBytes((long) Float.BYTES * numberDimensions);
                continue;
            }
            input.readBytes(binaryCode, 0, binaryCode.length);
            input.readFloats(correctiveValues, 0, correctiveValues.length);
            int quantizedComponentSum = input.readShort() & 0xffff;
            float score = score(
                numberDimensions,
                similarityFunction,
                query,
                queryCorrections,
                binaryCode,
                correctiveValues,
                quantizedComponentSum,
                centroidDp
            );
            knnCollector.incVisitedCount(1);
            knnCollector.collect(docId, score);
            if (knnCollector.earlyTerminated()) {
                return true;
            }
        }
        return false;
    }

    private static float score(
        int numberDimensions,
        VectorSimilarityFunction similarityFunction,
        byte[] quantizedQuery,
        OptimizedScalarQuantizer.QuantizationResult queryCorrections,
        byte[] binaryCode,
        float[] correctiveValues,
        int quantizedComponentSum,
        float centroidDp
    ) {
        float qcDist = ESVectorUtil.ipByteBinByte(quantizedQuery, binaryCode);
        float x1 = quantizedComponentSum;
        float ax = correctiveValues[0];
        // Here we assume `lx` is simply bit vectors, so the scaling isn't necessary
        float lx = correctiveValues[1];
        float ay = queryCorrections.lowerInterval();
        float ly = (queryCorrections.upperInterval() - ay) * FOUR_BIT_SCALE;
        float y1 = queryCorrections.quantizedComponentSum();
        float score = correctiveValues[2] + ax * ay * numberDimensions + ax * ly * y1 + ay * lx * x1 + lx * ly * qcDist;
        // For euclidean, we need to invert the score and apply the additional correction, which is
        // assumed to be the squared l2norm of the centroid centered vectors.
        if (similarityFunction == EUCLIDEAN) {
            score = queryCorrections.additionalCorrection() - 2f * score;
            return Math.max(1f / (1f + score), 0f);
        } else {
            // For cosine and max inner product, we need to apply the additional correction, which is
            // assumed to be the non-centered dot-product between the vector and the centroid
            score += queryCorrections.additionalCorrection() - centroidDp;
            if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
                return VectorUtil.scaleMaxInnerProductScore(score);
            }
            return Math.max((1f + score) / 2f, 0f);
        }
    }
}
