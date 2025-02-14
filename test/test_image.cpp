#include <gtest/gtest.h>
#include "image.hpp" 
#include <iostream>

// test for the zero constructor
TEST(ImageTest, ZeroConstructor){
    Image image(3, 3);
    
    // check if dimensions are correct
    EXPECT_EQ(image.nrows, 3);
    EXPECT_EQ(image.ncols, 3);

    // check if first and last elements are zero
    EXPECT_FLOAT_EQ(image(0, 0).x, 0.0f);
    EXPECT_FLOAT_EQ(image(0, 0).y, 0.0f);
    EXPECT_FLOAT_EQ(image(2, 2).x, 0.0f);
    EXPECT_FLOAT_EQ(image(2, 2).y, 0.0f);
}

// test for the accessor
TEST(ImageTest, Accessor){
    Image image(3, 3);
    
    // set some values
    image(0, 0).x = 1.0f;
    image(1, 1).x = 2.0f;
    image(2, 2).x = 3.0f;

    // check if values are correct
    EXPECT_FLOAT_EQ(image(0, 0).x, 1.0f);
    EXPECT_FLOAT_EQ(image(1, 1).x, 2.0f);
    EXPECT_FLOAT_EQ(image(2, 2).x, 3.0f);
}

// test the constructor from a .npy file with 4 byte floats
TEST(ImageTest, NpyConstructor4){
    // assuming we have a file test_image_2x3.npy with the shape (2, 3)
    Image image = Image("test_image_2x3_f4.npy");

    // Test that the data is correctly loaded
    EXPECT_EQ(image.nrows, 2);
    EXPECT_EQ(image.ncols, 3);
    
    EXPECT_EQ(image(0, 0).x, 11.0f);  
    EXPECT_EQ(image(0, 1).x, 12.0f); 
    EXPECT_EQ(image(0, 2).x, 13.0f); 
    EXPECT_EQ(image(1, 0).x, 21.0f); 
    EXPECT_EQ(image(1, 1).x, 22.0f); 
    EXPECT_EQ(image(1, 2).x, 23.0f); 
}


// test the constructor from a .npy file with 8 byte floats
TEST(ImageTest, NpyConstructor8){
    // assuming we have a file test_image_2x3.npy with the shape (2, 3)
    Image image = Image("test_image_2x3_f8.npy");

    // Test that the data is correctly loaded
    EXPECT_EQ(image.nrows, 2);
    EXPECT_EQ(image.ncols, 3);

    EXPECT_EQ(image(0, 0).x, 11.0f);  
    EXPECT_EQ(image(0, 1).x, 12.0f); 
    EXPECT_EQ(image(0, 2).x, 13.0f); 
    EXPECT_EQ(image(1, 0).x, 21.0f); 
    EXPECT_EQ(image(1, 1).x, 22.0f); 
    EXPECT_EQ(image(1, 2).x, 23.0f); 
}


int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
