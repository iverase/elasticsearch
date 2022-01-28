/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.geometry;

import java.util.Locale;

public enum Orientation {
    LEFT,
    RIGHT;

    public static final Orientation CLOCKWISE = Orientation.LEFT;
    public static final Orientation COUNTER_CLOCKWISE = Orientation.RIGHT;
    public static final Orientation CW = Orientation.LEFT;
    public static final Orientation CCW = Orientation.RIGHT;

    public boolean getAsBoolean() {
        return this == Orientation.RIGHT;
    }

    public static Orientation fromString(String orientation) {
        orientation = orientation.toLowerCase(Locale.ROOT);
        return switch (orientation) {
            case "right", "counterclockwise", "ccw" -> Orientation.RIGHT;
            case "left", "clockwise", "cw" -> Orientation.LEFT;
            default -> throw new IllegalArgumentException("Unknown orientation [" + orientation + "]");
        };
    }
}
