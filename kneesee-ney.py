#trigrams probability
#kneeserney implementation
import nltk
ngrams = nltk.trigrams("What a piece of work is man! how noble in reason! how infinite in faculty! in \
form and moving how express and admirable! in action how like an angel! in apprehension how like a god! \
the beauty of the world, the paragon of animals!")
freq_dist = nltk.FreqDist(ngrams)
kneser_ney = nltk.KneserNeyProbDist(freq_dist)
prob_sum = 0
for i in kneser_ney.samples():
prob_sum += kneser_ney.prob(i)
print(prob_sum)
