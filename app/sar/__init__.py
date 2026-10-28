"""Search planning core: a probability map of where the missing person is, drones that search it, and the update after each look.

Conventions: the search area is a raster of square cells indexed (row, col); positions are metres from the
top-left corner as (y, x); times are seconds since the search started. Probability outside the area is kept
as a single "rest of world" (ROW) value so the map never claims more certainty than it has.
"""
