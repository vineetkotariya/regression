import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE


    file = open("linreg_test.csv","r")
    f = []
    #Repeat for each song in the text file
    for line in file:
      i = 0
      #Let's split the line into an array called "fields" using the ";" as a separator:
      fields = line.split(",")
      f.append(fields)
      i = i + 1
      #and let's extract the data:
      #train
      #songTitle = fields[0]

    #It is good practice to close the file at the end to free up resources   
    file.close()

    f[0][1:] = [float(i) for i in f[0][1:]]
    f[1][1:] = [float(i) for i in f[1][1:]]
    
    test_area = np.asarray(f[0][1:])
    #print(test_area)
    test_price = np.asarray(f[1][1:])
    #print(test_price)

    file = open("linreg_train.csv","r")
    g = []
    #Repeat for each song in the text file
    for line in file:
      i = 0
      #Let's split the line into an array called "fields" using the ";" as a separator:
      fields = line.split(",")
      g.append(fields)
      i = i + 1
      #and let's extract the data:
      #train
      #songTitle = fields[0]

    #It is good practice to close the file at the end to free up resources   
    file.close()

    g[0][1:] = [float(i) for i in g[0][1:]]
    g[1][1:] = [float(i) for i in g[1][1:]]
    
    train_area = np.asarray(g[0][1:])
    print(train_area.shape)
    train_price = np.asarray(g[1][1:])
    #print(train_price)

    from scipy import stats

    slope, intercept, _, _, _ = stats.linregress(train_area, train_price)

    test_price = test_area

    val_price = area

    for i in range(test_area.size):
      test_price[i] = slope*test_area[i] + intercept

    for i in range(area.size):
      val_price[i] = slope*area[i] + intercept

    return val_price

    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
