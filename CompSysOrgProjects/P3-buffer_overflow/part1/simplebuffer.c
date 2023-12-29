#include <stdio.h>
#include <limits.h>
#include <unistd.h>

char* helper() {
  char buff[4];
  char* res = fgets(buff, INT_MAX, stdin); //INT_MAX = largest representable positive integer in 2's comp
  printf("echo:%.4s\nLocatedAt:%p\n",buff,&buff);
  return res;
}

int main(char* argv) {
  setreuid(1004,1004);
  helper();
  return 0;
}
