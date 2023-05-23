def linear_svc(text_to_predict):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn import metrics

    df = pd.read_csv('/home/mustafa7egazi/PycharmProjects/pythonProject/dataset.csv')

    x = df['text']
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Linear SVC:
    text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

    # teaching the algorithm
    text_clf_lsvc.fit(x_train, y_train)
    # acquire algorithm predictions
    predictions = text_clf_lsvc.predict(x_test)

    # acquire confusion matrix

    sub_index = []

    for i in y:
        if i not in sub_index:
            sub_index.append(i)

    print(pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=sub_index, columns=sub_index))

    # make a summary
    print(metrics.classification_report(y_test, predictions))

    print(metrics.accuracy_score(y_test, predictions))

    # prediction for unknown data

    user_disease = text_clf_lsvc.predict([text_to_predict])
    print(user_disease)
    return user_disease
