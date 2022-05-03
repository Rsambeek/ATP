#include "hwlib.hpp"

extern "C" int odd(int a);
extern "C" int even(int a);
extern "C" int sommig(int a);

int main(int argc, char const *argv[]) {
    while (1) {
        hwlib::cout << "odd(2): " << odd(2) << " | odd(5): " << odd(5) << " | even(2): " << even(2) << " | even(5): " << even(5) << " | sommig(6): " << sommig(6) << "\n";

        hwlib::wait_ms(500);
    }
}
