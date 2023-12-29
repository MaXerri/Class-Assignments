#include "server.h"

void vulnerable(char* secret, char* guess) {
  char outstr[MAXLINE];
  if (!strcmp(secret, guess)) {
    snprintf(outstr, MAXLINE, corAns);
    SEND_TO_CLIENT(outstr);
  } else {
    snprintf(outstr, MAXLINE, guess);
    SEND_TO_CLIENT(outstr);
    snprintf(outstr, MAXLINE, wrongAns);
    SEND_TO_CLIENT(outstr);      
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Missing argument for secret file\n");
    return 1;
  }
  //ALLOCATE STACK BUFFERS FOR USER INPUT AND SECRET MESSAGE
  char buffer[MAXLINE];  
  char secret[SECRET_SIZE];
  //READ THE SECRET MESSAGE FROM FILESYSTEM
  FILE *secretfile = fopen(argv[1], "r");
  fgets(secret, SECRET_SIZE, secretfile);

  //---Begin Socket Initializtion Code---/// (IGNORE THIS!!)
  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
  {
    perror("socket creation failed");
    exit(EXIT_FAILURE);
  }
  memset(&servaddr, 0, sizeof(servaddr));
  memset(&cliaddr, 0, sizeof(cliaddr));
  servaddr.sin_family = AF_INET; // IPv4
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(PORT);
  
  if (bind(sockfd, (const struct sockaddr *)&servaddr,
           sizeof(servaddr)) < 0)
  {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }
  //---End Socket Initializtion Code---// (OK STOP IGNORING!)


  
  while (1) {
    //Read from User
    int n = recvfrom(sockfd, (char *)buffer, MAXLINE,
		     MSG_WAITALL, (struct sockaddr *)&cliaddr,
		     &len);
    buffer[n] = '\0';
    //Reply to User
    vulnerable(secret, buffer);
  }
  return 0;
}
