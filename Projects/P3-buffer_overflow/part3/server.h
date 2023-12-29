#ifndef _SERVER_H
#define _SERVER_H
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <netinet/in.h>

#define PORT 8080
#define MAXLINE 1024
#define BUFFER_SIZE 128
#define SECRET_SIZE 128

int sockfd;
struct sockaddr_in servaddr, cliaddr;
int len = sizeof(cliaddr);
const char *corAns = "You guessed the secret! Good job :)\n";
const char *wrongAns = "\nWas Wrong!\n";

#define SEND_TO_CLIENT(msg) sendto(sockfd, (const char*) msg, strlen(outstr), MSG_CONFIRM, (const struct sockaddr *) &cliaddr, len)

#endif
