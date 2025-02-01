#include <gtest/gtest.h>
#include "image.hpp" 

// test for the zero constructor
TEST(ImageTest, ZeroConstructor){
    Image image(3, 3);
    
    // check if dimensions are correct
    EXPECT_EQ(image.nrows, 3);
    EXPECT_EQ(image.ncols, 3);

    // check if first and last elements are zero
    EXPECT_FLOAT_EQ(image(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(image(2, 2), 0.0f);
}

// test for the accessor
TEST(ImageTest, Accessor){
    Image image(3, 3);
    
    // set some values
    image(0, 0) = 1.0f;
    image(1, 1) = 2.0f;
    image(2, 2) = 3.0f;

    // check if values are correct
    EXPECT_FLOAT_EQ(image(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(image(1, 1), 2.0f);
    EXPECT_FLOAT_EQ(image(2, 2), 3.0f);
}

// // Test the constructor from a file (load from .npz)
// TEST(ImageTest, LoadFromFile) {
//     // Assuming you have a file test_image.npz with the shape (2, 2)
//     Image img = Image::load_from_file("test_image.npz");

//     // Test that the data is correctly loaded
//     EXPECT_EQ(img.nrows, 2);
//     EXPECT_EQ(img.ncols, 2);
//     EXPECT_EQ(img(0, 0), 1.0f); // Expect value in (0, 0) to be 1.0
//     EXPECT_EQ(img(1, 1), 2.0f); // Expect value in (1, 1) to be 2.0
// }

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
