function accuracy(true_labels, predicted_labels)
    return sum(true_labels .== predicted_labels) / length(predicted_labels)
end