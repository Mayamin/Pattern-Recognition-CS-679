
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;
// #include <random>

 /* ranf() is uniform in 0..1 */


double ranf(double m) {
 return (m*rand()/(double)RAND_MAX);
}
//extern float ranf(); 

float box_muller(float m, float s)      /* normal random variate generator */
{                                       /* mean m, standard deviation s */
        float x1, x2, w, y1;
        static float y2;
        static int use_last = 0;

        if (use_last)                   /* use value from previous call */
        {
                y1 = y2;
                use_last = 0;
        }
        else
        {
                do {
                        x1 = 2.0 * ranf(1.0) - 1.0;
                        x2 = 2.0 * ranf(1.0) - 1.0;
                        w = x1 * x1 + x2 * x2;

                        // x1 = 2.0 * ( rand() / RAND_MAX ) - 1.0;
                        // x2 = 2.0 * ( rand() / RAND_MAX ) - 1.0;
                        // w = x1 * x1 + x2 * x2;
                        printf("%f, ", w);
                } while ( w >= 1.0 );

                w = sqrt( (-2.0 * log( w ) ) / w );
                y1 = x1 * w;
                y2 = x2 * w;
                use_last = 1;
        }

        return( m + y1 * s );
}

// float not_box_muller(float m, float s)
// {
//     //These couple lines are how c++ creates a normal distribution so this will be used for the noise
//     //The normal distribution is created with mean .5 and standard deviation .1
//     std::random_device rd{};
//     std::mt19937 gen{rd()};
//     //double noise_mean = .5;
//   //  double noise_std = .1;
//     std::normal_distribution<> d{m, s};

//     return d(gen);
// }

int main (int argc, char** argv)
{
    
    // dataset A
    fstream fout_1;
    fout_1.open("dataset_A.csv", ios::out);

    // mew = 1 , sigma =1, label = 0
    for (int i = 0; i < 60000; i++)
    {
        
        // writing x value, y value and label value to file
        fout_1<< box_muller(1.0, 1.0)<< "," << box_muller(1.0, 1.0) << "," << "0" << endl;

    }

    // mew = 4 , sigma =1, label = 1
    for (int i = 0; i < 140000; i++)
    {
        
        fout_1<< box_muller(4.0, 1.0) << "," << box_muller(4.0, 1.0)<< "," << "1" << endl;

    }

    fout_1.close();
    //fout_2.close();

    printf("\n");


// dataset B
    
  fstream fout_2;
  fout_2.open("dataset_B.csv", ios::out);
    
    

for (int i = 0; i < 60000; i++)
    {
        
        fout_2<< box_muller(1.0, 1.0)<< "," << box_muller(1.0, 1.0) << "," << "0" << endl;
   
    }

    // mew = 4 , sigma =1, label = 1
    for (int i = 0; i < 140000; i++)
    {
        
        fout_2<< box_muller(4.0, 2.0) << "," << box_muller(4.0, 2.8) << "," << "1" << endl;
     
    }

    fout_1.close();
    fout_2.close();

    printf("\n");





    return 0;
}