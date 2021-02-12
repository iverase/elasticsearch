/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.spatial.vectortile;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.util.BitUtil;
import org.elasticsearch.common.geo.GeoPoint;
import org.elasticsearch.index.fielddata.LeafGeoPointFieldData;
import org.elasticsearch.index.fielddata.MultiGeoPointValues;
import org.elasticsearch.index.fielddata.plain.AbstractLatLonPointIndexFieldData;
import org.elasticsearch.xpack.spatial.proto.GeomCmd;
import org.elasticsearch.xpack.spatial.proto.GeomCmdHdr;
import org.elasticsearch.xpack.spatial.proto.VectorTile;
import org.locationtech.jts.geom.Envelope;

import java.util.ArrayList;
import java.util.List;

public class PointDocValuesVectorTileCollector extends AbstractVectorTileCollector {

    final AbstractLatLonPointIndexFieldData.LatLonPointIndexFieldData points;

    PointDocValuesVectorTileCollector(final AbstractLatLonPointIndexFieldData.LatLonPointIndexFieldData points, Envelope tileEnvelope, String field) {
        super(tileEnvelope, field);
        this.points = points;
    }

    @Override
    public VectorTileLeafCollector getVectorTileLeafCollector(LeafReaderContext context) {
        final double xScale = 1d / (tileEnvelope.getWidth() / (double) 4096);
        final double yScale = -1d / (tileEnvelope.getHeight() / (double) 4096);
        LeafGeoPointFieldData data = points.load(context);
        MultiGeoPointValues values = data.getGeoPointValues();
        final VectorTile.Tile.Feature.Builder featureBuilder = VectorTile.Tile.Feature.newBuilder();
        List<Integer> commands = new ArrayList<>();
        return (docID -> {
            featureBuilder.clear();
            commands.clear();
            featureBuilder.setType(VectorTile.Tile.GeomType.POINT);
            values.advanceExact(docID);
            GeoPoint point = values.nextValue();
            int x = (int) (xScale * (VectorTileUtils.lonToSphericalMercator(point.lon()) - tileEnvelope.getMinX()));
            int y = (int) (yScale * (VectorTileUtils.latToSphericalMercator(point.lat()) - tileEnvelope.getMinY())) + 4096;
            commands.add(GeomCmdHdr.cmdHdr(GeomCmd.MoveTo, 1));
            commands.add(BitUtil.zigZagEncode(x));
            commands.add(BitUtil.zigZagEncode(y));
            featureBuilder.addAllGeometry(commands);
            return featureBuilder;
        });
    }
}
