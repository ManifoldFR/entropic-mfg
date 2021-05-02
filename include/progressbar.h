#include <iostream>


namespace utils
{

#define RESET   "\033[0m"
#define BLUE    "\033[96m"

struct ProgressBar
{
    float progress = 0.;
    size_t ns;

    ProgressBar(size_t ns) : ns(ns) {
        print_progress();
    };

    void print_progress(size_t barWidth = 75)
    {
        std::cout << BLUE << "[";
        size_t pos = barWidth * progress;
        for (size_t i = 0; i  < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << RESET << size_t(progress * 100.) << " %\r";
        std::cout.flush();
        if (progress >= 1.0) {
            std::cout << std::endl;
        }
        progress += 1. / ns;
    }
};

}
