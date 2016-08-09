/*
   Holds a genotype, which consists of a) node types b) genes c) fitness
*/

#include "genotype.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>



static int genotype_size = -1; //default genotype size

//returns the size of genes
int genotype_get_size() {
  return genotype_size;
}

// setting a size of a genotype (should be without the size of genes)
void genotype_set_size(int size) {
  genotype_size = size;
}

// creating a genotype
Genotype genotype_create() {
  Genotype gen = malloc(sizeof(struct _Genotype_));
  gen->fitness = 0.0;
  gen->genes = malloc(genotype_size * sizeof(double));
  gen->node_types = malloc((int)sqrt(genotype_size) * sizeof(double));

  int i;
  for (i = 0; i < genotype_size; i++)
    gen->genes[i] = 0.0;
    
  for(i = 0; i < (int)sqrt(genotype_size); i++)
    gen->node_types[i] = 0.0;
    
  return gen;
}

// freeing a genotype
void genotype_destroy(Genotype g) {
  free(g->genes);
  free(g->node_types);
  free(g);
}

double drand(){
  double dr;
  int r;
  
  r = rand();
  dr = (double)(r)/(double)(RAND_MAX);
  return(dr);
}

// setting fitness to a genotype
void genotype_set_fitness(Genotype g, double fitness) {
  g->fitness = fitness;
}

// returns fitness fo a genotype
double genotype_get_fitness(Genotype g) {
  return g->fitness;
}

// returns genes of a genotype
const double *genotype_get_genes(Genotype *g) {
  return (*g)->genes;
}

// returns genotype node types
const double *genotype_get_node_types(Genotype *g){
  return (*g)->node_types;
}

//reads genotype from a file
int genotype_fread(Genotype *g, FILE *fd, int node_types_specified) {
  int i, result;

  
  result = 1;




  for (i = 0; i < (int)sqrt(genotype_size); i++) {
char x[1024];
int ret = 0;
    while (fscanf(fd, " %1023s", x) == 1) {
        //printf("%s",x);
ret = 1;
(*g)->node_types[i] = (float)atof(x);
//printf(" %f ",g->node_types[i]);
break;
    }



   // int ret = node_types_specified ? fscanf(fd, "%1023s", x) : 0.0; //0.0 default node for now
//printf("%s",x);
//g->node_types[i] = x
    if (ret == EOF)
      //fprintf(stderr, "Cannot decode the genotype file 1\n");
      result = 0;
  }
//printf("\n");
  
  //for (i = 0; i < genotype_size; i++) {
  //  int ret = fscanf(fd, "%1023s", &g->genes[i]);
  //  if (ret == EOF){
  //    //fprintf(stderr, "Cannot decode the genotype file 2\n");
  //    result = 0;
  //  }
  //}


  result = 1;




  for (i = 0; i < genotype_size; i++) {
char x[1024];
int ret = 0;
    while (fscanf(fd, " %1023s", x) == 1) {
        //printf("%s",x);
ret = 1;
(*g)->genes[i] = (float)atof(x);
//printf(" %f ",g->genes[i]);
break;
    }



   // int ret = node_types_specified ? fscanf(fd, "%1023s", x) : 0.0; //0.0 default node for now
//printf("%s",x);
//g->node_types[i] = x
    if (ret == EOF)
      //fprintf(stderr, "Cannot decode the genotype file 1\n");
      result = 0;
  }
//printf("\n");
  
  return result;
}

// writes genotype to a file
void genotype_fwrite(Genotype g, FILE *fd) {
  int i;
  for (i = 0; i < genotype_size; i++)
    fprintf(fd, " %lf", g->genes[i]);
    
  fprintf(fd, "\n");
}

// prints a genotype
void print_genotype(Genotype *g) {
  int i;
  printf("size is: %d\n", genotype_size);
  for (i = 0; i < genotype_size; i++)
    printf(" %f", (*g)->genes[i]);
    
  printf("\n");
}

