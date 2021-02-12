/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.spatial.vectortile;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.elasticsearch.xpack.spatial.proto.VectorTile;
import org.locationtech.jts.geom.Envelope;

import java.io.IOException;

public abstract class AbstractVectorTileCollector implements ShardVectorTileBuilder.VectorTileCollector {
    protected final String field;

    final Envelope tileEnvelope;

    final VectorTile.Tile.Layer.Builder layerBuilder = VectorTile.Tile.Layer.newBuilder();

    AbstractVectorTileCollector(Envelope tileEnvelope, String field) {
        this.field = field;
        this.tileEnvelope = tileEnvelope;
        layerBuilder.setVersion(2);
        layerBuilder.setName(field);
        layerBuilder.setExtent(4096);
    }

    @Override
    public byte[] getVectorTile() {
        final VectorTile.Tile.Builder tileBuilder = VectorTile.Tile.newBuilder();
        // Build MVT layer
        final VectorTile.Tile.Layer layer = layerBuilder.build();
        // Add built layer to MVT
        tileBuilder.addLayers(layer);
        /// Build MVT
        return tileBuilder.build().toByteArray();
    }

    protected interface VectorTileLeafCollector {
        VectorTile.Tile.Feature.Builder geometry(int docID) throws IOException;
    }

    public abstract VectorTileLeafCollector getVectorTileLeafCollector(LeafReaderContext context);


    @Override
    public LeafCollector getLeafCollector(LeafReaderContext context) {
        VectorTileLeafCollector collector = getVectorTileLeafCollector(context);

        return new LeafCollector() {

            @Override
            public void setScorer(Scorable scorer) {
            }

            @Override
            public void collect(int docID) throws IOException {
                layerBuilder.addFeatures(collector.geometry(docID));
            }
        };
    }

    @Override
    public ScoreMode scoreMode() {
        return ScoreMode.COMPLETE_NO_SCORES;
    }
}
