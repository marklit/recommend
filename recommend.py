#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collaborative Filtering ALS Recommender System using Spark MLlib adapted from
the Spark Summit 2014 Recommender System training example.

Usage:
    ./recommend.py train <training_data_file> [--partitions=<n>]
                   [--ranks=<n>] [--lambdas=<n>] [--iterations=<n>]
    ./recommend.py recommend <training_data_file> <movies_meta_data>
    ./recommend.py metrics <training_data_file> <movies_meta_data>
    ./recommend.py (-h | --help)

Options:
    -h, --help         Show this screen and exit.
    --partitions=<n>   Partition count [Default: 4]
    --ranks=<n>        Partition count [Default: 6,8,12]
    --lambdas=<n>      Partition count [Default: 0.1,1.0,10.0]
    --iterations=<n>   Partition count [Default: 10,20]

Examples:
    bin/spark-submit --driver-memory 2g recommend.py train ratings.dat
    bin/spark-submit recommend.py metrics ratings.dat movies.dat

Credits:

    Forked from:
        https://gist.github.com/rezsa/359714b3c9e0f554f878

    Comments in code derived from:
        http://blog.rezsa.com/2014/11/building-big-data-machine-learning_10.html
        http://spark.apache.org/docs/latest/programming-guide.html
        http://ampcamp.berkeley.edu/big-data-mini-course/movie-recommendation-with-mllib.html
        http://ampcamp.berkeley.edu/5/exercises/movie-recommendation-with-mllib.html
"""

import itertools
from math import sqrt
from operator import add
import sys

from docopt import docopt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS


SPARK_EXECUTOR_MEMORY = '2g'
SPARK_APP_NAME = 'movieRecommender'
SPARK_MASTER = 'local'


def parse_rating(line):
    """
    Parses a rating record that's in MovieLens format.
    
    :param str line: userId::movieId::rating::timestamp
    """
    fields = line.strip().split("::")

    # The data is divided into three parts for training, validation, and
    # testing. This is why the sets were keyed with integers < 10. These 
    # methods are very quick and scalable big-data tricks to make random 
    # key-value buckets without using any randomizing functions.
    return long(fields[3]) % 10, (int(fields[0]), # User ID
                                  int(fields[1]), # Movie ID
                                  float(fields[2])) # Rating


def compute_rmse(model, data, validation_count):
    """
    Compute RMSE (Root Mean Squared Error).

    :param object model:
    :param list data:
    :param integer validation_count:
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = \
        predictions.map(lambda x: ((x[0], x[1]), x[2])) \
                   .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
                   .values()
    return sqrt(
        predictionsAndRatings.map(
            lambda x: (x[0] - x[1]) ** 2
        ).reduce(add) / float(validation_count)
    )


def metrics(training_data_file, movies_meta_data):
    """
    Print metrics for the ratings database
    
    :param str training_data_file: file location of ratings.dat
    :param str movies_meta_data: file location of movies.dat
    """
    conf = SparkConf().setMaster(SPARK_MASTER) \
                      .setAppName(SPARK_APP_NAME) \
                      .set("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
    spark_context = SparkContext(conf=conf)

    movies = {}

    with open(movies_meta_data, 'r') as open_file:
        movies = {int(line.split('::')[0]): line.split('::')[1]
                  for line in open_file
                  if len(line.split('::')) == 3}

    # The training file with all the rating is loaded as a spark Resilient 
    # Distributed Dataset (RDD), and the parse_rating method is applied to
    # each line that has been read from the file. RDD is a fault-tolerant 
    # collection of elements that can be operated on in parallel.
    ratings = spark_context.textFile(training_data_file) \
                           .filter(lambda x: x and len(x.split('::')) == 4) \
                           .map(parse_rating)

    most_rated = ratings.values() \
                        .map(lambda r: (r[1], 1)) \
                        .reduceByKey(add) \
                        .map(lambda r: (r[1], r[0])) \
                        .sortByKey(ascending=False) \
                        .collect()[:10]

    spark_context.stop()

    print
    print '10 most rated films:'

    for (ratings, movie_id) in most_rated:
        print '{:10,} #{} {}'.format(ratings, movie_id, movies[movie_id])


def train(training_data_file, numPartitions, ranks, lambdas, numIters):
    """
    Print metrics for the ratings database
    
    :param str training_data_file: file location of ratings.dat
    :param int numPartitions: number of partitions
    :param list ranks: list of ranks to use
    :param list lambdas: list of lambdas to use
    :param list numIters: list of iteration counts
    """
    conf = SparkConf().setMaster(SPARK_MASTER) \
                      .setAppName(SPARK_APP_NAME) \
                      .set("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
    spark_context = SparkContext(conf=conf)

    # The training file with all the rating is loaded as a spark Resilient 
    # Distributed Dataset (RDD), and the parse_rating method is applied to
    # each line that has been read from the file. RDD is a fault-tolerant 
    # collection of elements that can be operated on in parallel.
    ratings = spark_context.textFile(training_data_file) \
                           .filter(lambda x: x and len(x.split('::')) == 4) \
                           .map(parse_rating)

    numRatings = ratings.count()

    numUsers = ratings.values() \
                      .map(lambda r: r[0]) \
                      .distinct() \
                      .count()

    numMovies = ratings.values() \
                       .map(lambda r: r[1]) \
                       .distinct() \
                       .count()

    training = ratings.filter(lambda x: x[0] < 6) \
                      .values() \
                      .repartition(numPartitions) \
                      .cache()

    validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
                        .values() \
                        .repartition(numPartitions) \
                        .cache()

    test = ratings.filter(lambda x: x[0] >= 8) \
                  .values() \
                  .cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    # We will test 18 combinations resulting from the cross product of 3 
    # different ranks (6, 8, 12), 3 different lambdas (0.1, 1.0, 10.0), 
    # and two different numbers of iterations (10, 20). We will use
    # compute_rmse to compute the RMSE (Root Mean Squared Error) on the
    # validation set for each model. The model with the smallest RMSE on the
    # validation set becomes the one selected and its RMSE on the test set is
    # used as the final metric.
    bestValidationRmse = float("inf")
    bestModel, bestRank, bestLambda, bestNumIter = None, 0, -1.0, -1

    # Collaborative filtering is commonly used for recommender systems. These 
    # techniques aim to fill in the missing entries of a user-item association
    # matrix, in our case, the user-movie rating matrix. MLlib currently
    # supports model-based collaborative filtering, in which users and products
    # are described by a small set of latent factors that can be used to
    # predict missing entries. In particular, we implement the alternating 
    # least squares (ALS) algorithm to learn these latent factors.
    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(ratings=training,
                          rank=rank,
                          iterations=numIter,
                          lambda_=lmbda)

        validationRmse = compute_rmse(model, validation, numValidation)

        if validationRmse < bestValidationRmse:
            bestModel, bestValidationRmse = model, validationRmse
            bestRank, bestLambda, bestNumIter = rank, lmbda, numIter

    # Evaluate the best model on the test set
    testRmse = compute_rmse(bestModel, test, numTest)

    spark_context.stop()

    print
    print 'Ratings:     {:10,}'.format(numRatings)
    print 'Users:       {:10,}'.format(numUsers)
    print 'Movies:      {:10,}'.format(numMovies)
    print
    print 'Training:    {:10,}'.format(numTraining)
    print 'Validation:  {:10,}'.format(numValidation)
    print 'Test:        {:10,}'.format(numTest)
    print
    print 'The best model was trained with:'
    print '    Rank:             {:10,}'.format(bestRank)
    print '    Lambda:           {:10,.6f}'.format(bestLambda)
    print '    Iterations:       {:10,}'.format(bestNumIter)
    print '    RMSE on test set: {:10,.6f}'.format(testRmse)


def recommend(training_data_file, movies_meta_data):
    """
    Print metrics for the ratings database
    
    :param str training_data_file: file location of ratings.dat
    :param str movies_meta_data: file location of movies.dat
    """
    conf = SparkConf().\
            setMaster("local").\
            setAppName("movieRecommender").\
            set("spark.executor.memory", "2g")
    spark_context = SparkContext(conf=conf)

    movies = {}

    with open(movies_meta_data, 'r') as open_file:
        movies = {int(line.split('::')[0]): line.split('::')[1]
                  for line in open_file
                  if len(line.split('::')) == 3}

    # The training file with all the rating is loaded as a spark Resilient 
    # Distributed Dataset (RDD), and the parse_rating method is applied to
    # each line that has been read from the file. RDD is a fault-tolerant 
    # collection of elements that can be operated on in parallel.
    ratings = spark_context.textFile(training_data_file) \
                           .filter(lambda x: x and len(x.split('::')) == 4) \
                           .map(parse_rating)

    my_ratings = [
        # User ID 0, Movie ID, Rating
        (0,          2858,     5.0), # American Beauty (1999)
        (0,          480,      4.0), # Jurassic Park (1993)
        (0,          589,      5.0), # Terminator 2: Judgement Day (1991)
        (0,          2571,     5.0), # Matrix, The (1999)
        (0,          1270,     3.0), # Back to the Future (1985)
    ]

    my_ratings_rdd = spark_context.parallelize(my_ratings, 1)

    # .union(my_ratings_rdd) after filter

    # Exclude films I've rated from the recommended list
    my_rated_films = set([rating[1]
                          for rating in my_ratings])

    # candidates = spark_context \
    #                 .parallelize([rating
    #                               for rating in ratings.values()
    #                               if rating[1][1] not in my_rated_films]) \
    #                 .repartition(numPartitions) \
    #                 .cache()

    # predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))) \
    #                        .collect()

    # recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

    # print
    # print 'Recommendations:'
    # from pprint import pprint
    # print pprint(recommendations)

    # for index in xrange(len(recommendations)):
    #     movie_id = recommendations[index][1]
    #     factor = recommendations[index][2]
    #     print movies[movie_id] if movie_id in movies else movie_id, \
    #           '{:10,.6f}'.format(factor)
    

def main(argv):
    """
    :param dict argv: command line arguments
    """
    opt = docopt(__doc__, argv)

    if opt['train']:
        ranks    = [int(rank)      for rank in opt['--ranks'].split(',')]
        lambdas  = [float(_lambda) for _lambda in opt['--lambdas'].split(',')]
        numIters = [int(_iter)     for _iter in opt['--iterations'].split(',')]

        train(opt['<training_data_file>'],
              int(opt['--partitions']),
              ranks,
              lambdas,
              numIters)

    if opt['metrics']:
        metrics(opt['<training_data_file>'],
                opt['<movies_meta_data>'])

    if opt['recommend']:
        recommend(opt['<training_data_file>'],
                  opt['<movies_meta_data>'])


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        pass
