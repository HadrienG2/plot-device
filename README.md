# Plot-device: A quick experiment in Vulkan-based plotting

As a Vulkan learning exercise, I made a small function plotter.

The underlying plotting algorithm is mathematically flawed (line width
overshoots near rapid derivative changes, as shown on the attached picture) and
should be scrapped in favor of a more classical polyline-based design if this
were turned into a serious plotting library. But there are already many of these
around, and personally I had seen what I wanted to see.