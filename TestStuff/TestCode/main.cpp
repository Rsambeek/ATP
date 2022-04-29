#include "hwlib.hpp"

extern "C" int odd();

int main(int argc, char const *argv[]) {
    while (1) {
        hwlib::cout << odd() << "\n";

        hwlib::wait_ms(500);
    }
}
