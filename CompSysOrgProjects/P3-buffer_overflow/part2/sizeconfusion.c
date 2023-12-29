#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

//Reads an integer from the standard input
//stops once it sees an EndOfFile or a newline character
int readInt() {
  char strbuff[10]; //big enough to represent any int
  int inchar = fgetc(stdin);
  int curChar = 0;
  while (curChar < 10 && inchar != EOF && inchar != ((int) '\n')) {
    strbuff[curChar] = (char) inchar;
    curChar += 1;
    inchar = fgetc(stdin);    
  }
  strbuff[curChar] = '\0';
  return atoi(strbuff);
}



//Read in a bunch of integers and store them in dest
int vulnerable(int* dest, uint len) {
  int reads = 0;
  if (!len) return 0;
  while (reads < len) {
    int count = readInt();
    if (count) {
      dest[reads] = count;
      reads += 1;      
    } else {
      //once we encounter an error or read zero we stop
      break;
    }
  }
  return reads;
}


//Ask the user how many integers to read (from stdin)
void main() {
  setreuid(1005, 1005);
  int toRead = readInt();
  int ibuff[toRead];
  printf("Successfully copied %d integers\n",vulnerable(ibuff, toRead));
  return;
}
