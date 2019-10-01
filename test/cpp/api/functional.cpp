#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

namespace F = torch::nn::functional;

using namespace torch::nn;

struct FunctionalTest : torch::test::SeedingFixture {};

TEST_F(FunctionalTest, MaxPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::max_pool1d(x, MaxPool1dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1 ,2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(FunctionalTest, MaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::max_pool2d(x, MaxPool2dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2 ,2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
}

TEST_F(FunctionalTest, MaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::max_pool3d(x, MaxPool3dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::avg_pool1d(x, AvgPool1dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 2}));
}

TEST_F(FunctionalTest, AvgPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::avg_pool2d(x, AvgPool2dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2}));
}

TEST_F(FunctionalTest, AvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::avg_pool3d(x, AvgPool3dOptions(3).stride(2));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 2, 2, 2}));
}

TEST_F(FunctionalTest, CosineSimilarity) {
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  auto output = F::cosine_similarity(input1, input2, CosineSimilarityOptions().dim(1));
  auto expected = torch::tensor({0.8078, 0.8721}, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected, 1e-04));
}

TEST_F(FunctionalTest, PairwiseDistance) {
  auto input1 = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat);
  auto input2 = torch::tensor({{1, 8, 3}, {2, 1, 6}}, torch::kFloat);
  auto output = F::pairwise_distance(input1, input2, PairwiseDistanceOptions(1));
  auto expected = torch::tensor({6, 6}, torch::kFloat);
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, PDist) {
  {
    auto input = torch::tensor({{-1.0, -5.0, -1.0}, {2.0, 4.0, 6.0}});
    auto output = F::pdist(input);
    auto expected = torch::tensor({11.7898});
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    auto input = torch::tensor({{1.0, -1.0}, {1.0, 3.0}, {3.0, 3.0}});
    auto output = F::pdist(input, 1.5);
    auto expected = torch::tensor({4.0, 4.8945, 2.0});
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(FunctionalTest, AdaptiveMaxPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::adaptive_max_pool1d(x, AdaptiveMaxPool1dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 3}));
}

TEST_F(FunctionalTest, AdaptiveMaxPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::adaptive_max_pool2d(x, AdaptiveMaxPool2dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveMaxPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::adaptive_max_pool3d(x, AdaptiveMaxPool3dOptions(3));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool1d) {
  auto x = torch::ones({1, 1, 5});
  auto y = F::adaptive_avg_pool1d(x, AdaptiveAvgPool1dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({1, 1, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool2d) {
  auto x = torch::ones({2, 5, 5});
  auto y = F::adaptive_avg_pool2d(x, AdaptiveAvgPool2dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3}));
}

TEST_F(FunctionalTest, AdaptiveAvgPool3d) {
  auto x = torch::ones({2, 5, 5, 5});
  auto y = F::adaptive_avg_pool3d(x, AdaptiveAvgPool3dOptions(3));

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(torch::allclose(y, torch::ones({2, 3, 3, 3})));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({2, 3, 3, 3}));
}

TEST_F(FunctionalTest, HingeEmbeddingLoss) {
  auto input = torch::tensor({{2, 22, 4}, {20, 10, 0}}, torch::kFloat);
  auto target = torch::tensor({{2, 6, 4}, {1, 10, 0}}, torch::kFloat);
  auto output = F::hinge_embedding_loss(
      input, target, HingeEmbeddingLossOptions().margin(2));
  auto expected = torch::tensor({10}, torch::kFloat);

  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(FunctionalTest, MaxUnpool1d) {
  auto x = torch::tensor({{{2, 4, 5}}}, torch::requires_grad());
  auto indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  auto y = F::max_unpool1d(x, indices, MaxUnpool1dOptions(3));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 9}));

  x = torch::tensor({{{2, 4, 5}}}, torch::requires_grad());
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  y = F::max_unpool1d(x, indices, MaxUnpool1dOptions(3), c10::IntArrayRef({1, 1, 9}));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 9}));

  x = torch::tensor({{{2, 4, 5}}}, torch::requires_grad());
  indices = torch::tensor({{{1, 3, 4}}}, torch::kLong);
  y = F::max_unpool1d(x, indices, MaxUnpool1dOptions(3).stride(2).padding(1));

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(torch::allclose(y, torch::tensor({{{0, 2, 0, 4, 5}}}, torch::kFloat)));
  ASSERT_EQ(y.sizes(), torch::IntArrayRef({1, 1, 5}));
}
