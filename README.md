# Requirements

```bash
sudo apt-get install -yq python-software-properties
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install -yq oracle-java7-installer
sudo apt-get install -yq scala default-jdk
java -version
javac -version

wget http://apache.cs.utah.edu/spark/spark-1.3.0/spark-1.3.0.tgz
tar xvf spark-1.3.0.tgz
cd spark-1.3.0/
build/sbt assembly # This takes around 30 minutes...

virtualenv spark
source spark/bin/activate
git clone https://github.com/marklit/recommend.git .
pip install -r requirements.txt
```

# Film ratings data

```bash
curl -O http://files.grouplens.org/papers/ml-1m.zip
unzip -j ml-1m.zip
```

# Example outputs

## Training

```bash
$ bin/spark-submit recommend.py train ratings.dat
```

```
Ratings:      1,000,209
Users:            6,040
Movies:           3,706

Training:       602,241
Validation:     198,919
Test:           199,049

The best model was trained with:
    Rank:                     12
    Lambda:             0.100000
    Iterations:               20
    RMSE on test set:   0.869235
```

```bash
$ bin/spark-submit recommend.py train ratings.dat \
    --ranks=8,9,10 --lambdas=0.31,0.32,0.33 --iterations=3
```

```
The best model was trained with:
    Rank:                     10
    Lambda:             0.320000
    Iterations:                3
    RMSE on test set:   0.931992
```

```bash
$ bin/spark-submit recommend.py train ratings.dat \
    --ranks=5,10,15,20 --lambdas=0.33,0.5,0.8,0.9 --iterations=3,6,9
```

```
The best model was trained with:
    Rank:                     15
    Lambda:             0.330000
    Iterations:                3
    RMSE on test set:   0.939317
```

## Recommending

```bash
$ bin/spark-submit recommend.py recommend ratings.dat movies.dat
```

```
His Girl Friday (1940)
New Jersey Drive (1995)
Breakfast at Tiffany's (1961)
Halloween 5: The Revenge of Michael Myers (1989)
Just the Ticket (1999)
I'll Be Home For Christmas (1998)
Goya in Bordeaux (Goya en Bodeos) (1999)
For the Moment (1994)
Thomas and the Magic Railroad (2000)
Message in a Bottle (1999)
Light of Day (1987)
...
```