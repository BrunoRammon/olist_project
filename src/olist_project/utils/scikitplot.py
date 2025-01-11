import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from numpy import interp
import numpy as np

def _roc(y_true, y_probas, title='ROC Curves',
         plot_micro=True, plot_macro=True, classes_to_plot=None,
         ax=None, figsize=None, cmap='nipy_spectral',
         title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.

        plot_macro (boolean, optional): Plot the macro average ROC curve.
            Defaults to ``True``.

        classes_to_plot (list-like, optional): Classes for which the ROC
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i],
                                                pos_label=classes[i])
        if to_plot:
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr_dict[i], tpr_dict[i], lw=2, color=color,
                    label=f'ROC curve of class {classes[i]} (area = {roc_auc:0.2f})')

    if plot_micro:
        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack(
                (1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label='micro-average ROC curve '
                      f'(area = {roc_auc:0.2f})',
                color='deeppink', linestyle=':', linewidth=4)

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr,
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax

def _binary_ks_curve(y_true, y_probas):
    """This function generates the points necessary to calculate the KS
    Statistic curve.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_probas (array-like, shape (n_samples)): Probability predictions of
            the positive class.

    Returns:
        thresholds (numpy.ndarray): An array containing the X-axis values for
            plotting the KS Statistic plot.

        pct1 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        pct2 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        ks_statistic (float): The KS Statistic, or the maximum vertical
            distance between the two curves.

        max_distance_at (float): The X-axis value at which the maximum vertical
            distance between the two curves is seen.

        classes (np.ndarray, shape (2)): An array containing the labels of the
            two classes making up `y_true`.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
            is only relevant in binary classification.
    """
    y_true, y_probas = np.asarray(y_true), np.asarray(y_probas)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError('Cannot calculate KS statistic for data with '
                         f'{len(lb.classes_)} category/ies')
    idx = encoded_labels == 0
    data1 = np.sort(y_probas[idx])
    data2 = np.sort(y_probas[np.logical_not(idx)])

    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
    while ctr1 < len(data1) or ctr2 < len(data2):

        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1

        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1

        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)

    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])

    differences = pct1 - pct2
    ks_statistic, max_distance_at = (np.max(differences),
                                     thresholds[np.argmax(differences)])

    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_

def _ks_statistic(y_true, y_probas, title='KS Statistic Plot',
                  ax=None, figsize=None, title_fontsize="large",
                  text_fontsize="medium"):
    """Generates the KS Statistic plot from labels and scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "KS Statistic Plot".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_ks_statistic(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_ks_statistic.png
           :align: center
           :alt: KS Statistic
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate KS statistic for data with '
                         f'{len(classes)} category/ies')
    probas = y_probas

    # Compute KS Statistic curves
    thresholds, pct1, pct2, ks_statistic, \
        max_distance_at, classes = _binary_ks_curve(y_true,
                                                    probas[:, 1].ravel())

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(thresholds, pct1, lw=3, label=f'Class {classes[0]}')
    ax.plot(thresholds, pct2, lw=3, label=f'Class {classes[1]}')
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(max_distance_at, *sorted([pct1[idx], pct2[idx]]),
               label=f'KS Statistic: {ks_statistic:.3f} at {max_distance_at:.3f}',
               linestyle=':', lw=3, color='black')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel('Threshold', fontsize=text_fontsize)
    ax.set_ylabel('Percentage below threshold', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax

def _precision_recall(y_true, y_probas,
                      title='Precision-Recall Curve',
                      plot_micro=True,
                      classes_to_plot=None, ax=None,
                      figsize=None, cmap='nipy_spectral',
                      title_fontsize="large",
                      text_fontsize="medium"):
    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Precision-Recall curve".

        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.

        classes_to_plot (list-like, optional): Classes for which the precision-recall
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_precision_recall(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    binarized_y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        binarized_y_true = np.hstack(
            (1 - binarized_y_true, binarized_y_true))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        if to_plot:
            average_precision = average_precision_score(
                binarized_y_true[:, i],
                probas[:, i])
            precision, recall, _ = precision_recall_curve(
                y_true, probas[:, i], pos_label=classes[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall, precision, lw=2,
                    label=f'Precision-recall curve of class {classes[i]} '
                          f'(area = {average_precision:0.3f})',
                    color=color)

    if plot_micro:
        precision, recall, _ = precision_recall_curve(
            binarized_y_true.ravel(), probas.ravel())
        average_precision = average_precision_score(binarized_y_true,
                                                    probas,
                                                    average='micro')
        ax.plot(recall, precision,
                label='micro-average Precision-recall curve '
                      f'(area = {average_precision:0.3f})',
                color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    return ax

def cumulative_gain_curve(y_true, y_score, pos_label=None):
    """This function generates the points necessary to plot the Cumulative Gain

    Note: This implementation is restricted to the binary classification task.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).

        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative

    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.

        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains

def _cumulative_gain(y_true, y_probas, title='Cumulative Gains Curve',
                     ax=None, figsize=None, title_fontsize="large",
                     text_fontsize="medium"):
    """Generates the Cumulative Gains Plot from labels and scores/probabilities

    The cumulative gains chart is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://mlwiki.org/index.php/Cumulative_Gain_Chart. The implementation
    here works only for binary classification.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Cumulative Gains Curve".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_cumulative_gain(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_cumulative_gain.png
           :align: center
           :alt: Cumulative Gains Plot
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         f'{len(classes)} category/ies')

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0],
                                                classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains1, lw=3, label=f'Class {classes[0]}')
    ax.plot(percentages, gains2, lw=3, label=f'Class {classes[1]}')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax

def _lift_curve(y_true, y_probas, title='Lift Curve',
                ax=None, figsize=None, title_fontsize="large",
                text_fontsize="medium"):
    """Generates the Lift Curve from labels and scores/probabilities

    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html.
    The implementation here works only for binary classification.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Lift Curve".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_lift_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_lift_curve.png
           :align: center
           :alt: Lift Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Lift Curve for data with '
                         f'{len(classes)} category/ies')

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0],
                                                classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])

    percentages = percentages[1:]
    gains1 = gains1[1:]
    gains2 = gains2[1:]

    gains1 = gains1 / percentages
    gains2 = gains2 / percentages

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains1, lw=3, label=f'Class {classes[0]}')
    ax.plot(percentages, gains2, lw=3, label=f'Class {classes[1]}')

    ax.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Lift', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax
