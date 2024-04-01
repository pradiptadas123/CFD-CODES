#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main() {
  int guess;
  int ran;

  srand(time(0));

  cout << "I am thinking of a number between 1 and 10." << endl;
  cout << "Can you guess it within 3 guesses?" << endl;

  ran = ((rand() % 9) + 1);

  while (guess != ran) {
    if (guess < 1 || guess > 10) {
      cout << "Please enter a guess between 1 and 10." << endl;
    } else {
      cout << "You lose! The number was " << ran << endl;
    }
    cin >> guess;
  }

  cout << "You win! The number was " << ran << endl;

  return 0;
}
