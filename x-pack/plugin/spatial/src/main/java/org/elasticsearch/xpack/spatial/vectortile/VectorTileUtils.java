/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.spatial.vectortile;

import org.elasticsearch.geometry.Rectangle;
import org.locationtech.jts.geom.Envelope;

public class VectorTileUtils {


    public static Envelope getJTSTileBounds(int x, int y, int zoom)
    {
        return new Envelope(getLong(x, zoom), getLong(x + 1, zoom), getLat(y, zoom), getLat(y + 1, zoom));
    }

    public static Rectangle getTileBounds(int x, int y, int zoom)
    {
        return new Rectangle(getLong(x, zoom), getLong(x + 1, zoom), getLat(y + 1, zoom), getLat(y, zoom));
    }

    public static double getLong(int x, int zoom)
    {
        return lonToSphericalMercator( x / Math.pow(2, zoom) * 360 - 180 );
    }

    public static double getLat(int y, int zoom)
    {
        double r2d = 180 / Math.PI;
        double n = Math.PI - 2 * Math.PI * y / Math.pow(2, zoom);
        return latToSphericalMercator(r2d * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n))));
    }

    public static double lonToSphericalMercator(double lon) {
        return lon * 20037508.34 / 180;
    }

    public static double latToSphericalMercator(double lat) {
        double y = Math.log(Math.tan((90 + lat) * Math.PI / 360)) / (Math.PI / 180);
        return y * 20037508.34 / 180;
    }
}
