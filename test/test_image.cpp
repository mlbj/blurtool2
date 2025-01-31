#include <gtest/gtest.h>
#include "image.hpp" 

// a basic test for the accessor
TEST(ImageTest, PixelAccess){
    Image image(5, 5);
    image(0, 1) = 1.0f;
    EXPECT_FLOAT_EQ(image(0, 1), 1.0f);
}

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
