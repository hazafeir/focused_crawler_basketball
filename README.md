# focused_crawler_basketball
A focused crawler about the theme "basketball" in greek typed links.

get_data_onsports_basket.py : get data from directory basketball onsports.gr to train classifiers

get_data_onsports_not_basket.py : get data from other directories instead basketball onsports.gr to train classifiers

get_data_to_10_basket.py : get data from directory basketball to10.gr to train classifiers

get_data_to_10_not_basket.py : get data from other directories instead basketball to10.gr to train classifiers

basket_dataset.csv : data set of all data (onsports.gr, to10.gr)

Basket_Classification.py : training of classifiers

NB_model.sav : model of naive bayes classifiers made in Basket_Classification.py

SVM_model.sav  : model of svm classifiers made in Basket_Classification.py

vectorizer.pickle : model of tfidf vectorization made in Basket_Classification.py

f_crawler1.py : focused crawler for onsports.gr (it gets link of a page as directories, so seed_url + href)

f_crawler2.py : focused crawler for sport24.gr
