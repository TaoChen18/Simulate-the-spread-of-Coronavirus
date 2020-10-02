#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main(){
  int rc;
  int p[2];
  char buffer[30];
  close(2);
  rc = pipe(p);
  printf("%d-%d-%d\n",getpid(),p[0],p[1]);
  rc = fork();
  if(rc == 0){
    rc = write(p[1],"WINTERSUMMERWINTERSUMMERWINTER",24);
    printf("%d-%d\n",getpid(),rc);
    rc = fork();
  }
  if(rc > 0){
    int n = p[1] + p[0] * p[0];
    rc = read(p[0],buffer,n);
    buffer[rc] = '\0';
    printf("%d-%s\n",getpid(),buffer);
  }else{
    wait(NULL);
    printf("%d-%d\n",getpid(),rc);
  }
  return 0;
}
