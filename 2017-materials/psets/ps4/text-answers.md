# 6. 7650 only: comparing hyperparameters (1 point)

Do a systematic comparison of one hyperparameter: could be input embedding size, stack embedding size, learning rate, dropout, or something you added for the bakeoff. Try at least five different values, using either your system from 4.4 or from the bakeoff. Explain what you tried and what you found.

I was going to test the stack embedding size parameter, on the values [16, 32, 64, 128, 500, 1000]. Unfortunately, I have encountered an issue while attempting to use Docker after a computer restart and have not been able to carry the experiment out. The issue is this one:
https://github.com/docker/docker/issues/30239
And the fix, here:
https://github.com/docker/docker/pull/31668
is not applicable to the only version of Docker that I was able to install on my laptop (docker toolbox).
My guess at the results would be that there is a maximum at the 128 setting, showing that too low a dimension prohibits the combiner network from providing enough meaningful information about the input words to the decision layer. If the dimension is too high, sparsity of the data comes to play, where the combiner network itself cannot "decide" which aspects of the word embeddings are significant for deciding the next action.
