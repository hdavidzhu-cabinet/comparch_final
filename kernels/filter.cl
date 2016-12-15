__constant sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

float getFilterValue(__constant const float* filterWeights, const int x, const int y) {
	return filterWeights[(x + FILTER_SIZE) + (y + FILTER_SIZE) * (FILTER_SIZE * 2 + 1)];
}

__kernel void filter(
  __read_only image2d_t input,
  __constant float* filterWeights,
  __write_only image2d_t output) {

  const int2 position = { get_global_id(0), get_global_id(1) };
  float4 sum = (float4)(0.0f);
  for(int y = -FILTER_SIZE; y <= FILTER_SIZE; y++) {
    for(int x = -FILTER_SIZE; x <= FILTER_SIZE; x++) {
      sum += getFilterValue(filterWeights, x, y) * read_imagef(input, sampler, position + (int2)(x, y));
    }
  }

  write_imagef(output, (int2)(position.x, position.y), sum);
}
