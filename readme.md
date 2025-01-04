binomial_distribution(n, k, p) = 
choose(n, k) * pow(p, k) * pow(1 - p, n - k)

hoeffding_inequality(v, u, e, N) =
2 * exp(-2 * (e ** 2) * N)

if u = 0.9, what is the probability that a sample of 10 marbles will have v <= 0.1

v <= 0.1 for a sample of size 10 => # of successes <= 1 for a sample of size 10

the probability is:
sum([binomial_distribution(10, k, .9) for k in range(2)])
= 1 * (0.9 ** 0) * (0.1 ** 10) + 10 * (0.9 ** 1) * (0.1 ** 9) = 9 * (10 ** -9) = .000000009

very unlikely to only get "success" outcome 0 or 1 times in a sample of size 10 when 90% of the outcomes are "successes" 

if u = 0.9, use the Hoeffding Inequality to bound the probability that a sample of 10 marbles will have v <= 0.1 and compare the answer to the previous exercise


since P(v <= 0.1) = P(v < 0.2) for a sample of size 10, e = 0.7

then |v - u| > 0.7 becomes v - u > 0.7 or v - u < - 0.7
which becomes v > 1.6 or v < 0.2 note after 1 probability is 0. (cannot have sample proportion bigger than 1)
so everything tracks.

hoeffding_inequality(v, u, e, N) =
2 * exp(-2 * (0.7 ** 2) * 10) = 1.10903 * (10 ** - 4)

this is about 10^5 = 100000 times bigger than the probability estimate so it is not as tight.


"noise" means that there are inputs mapped to multiple (binary) outputs.
the deterministic target will predict y + some error do to the y values that it can't predict due to there being more than 1 output for the same input. we can partition the tuples so that  one set are those for which
f and y agree and the other where they disagree. since we know out of sample error is mu for h we can get two cases for the error h makes with y
event 1) ex. h(x) != y and f(x) = y

event 2) ex. h(x) != y and f(x) != y


I know P(y | x) = [lamda y = f(x), 1 - lambda y != f(x)]
P(h != f) = mu for f not noisy

so P(h(x) != y) = P((h(x) != y and f(x) = y) or (h(x) != y and f(x) != y))
= P(h(x) != y given f(x) = y) * P(f(x) = y) + P(h(x) != y given f(x) != y) * P(f(x) != y)
= mu * lambda + (1 - mu) * (1 - lambda)
= 1 + 2* mu * lamda -(mu + lambda)

when lambda = 0.5 this becomes

1 + mu - mu - 0.5 = 0.5

this makes sense because if lambda = 0.5 f classifies the inputs with equal probability as both outcomes so if f doesn't tell me anything about x then h which is a knockoff of f can't either

looking at the graph error with noisy target will be minimal when out o sample error is minimal and target is quiet (deterministic with respect to f, f can for sure give me y when applied to x). error with noisy target will also be minimal when when out o sample error is maximal and target is "quiet" (in the sense that f can for sure give me the opposite of y when applied to x which is a dual situation probably the same as the first situation)
error will be maximized when h fails completely to perform out of same (out of sample error probability is 1) and target is quiet (y completely determined by f)
or when h completely agrees with f (out of sample error is 0) but f can only tell me the opposite of y (dual mathematically but doesn't make sense I guess)

as the target y gets noisier for a given out of sample error target error increases

problem 1.4
for training set of size 20, the perceptron converged within 20 updates with one outlier within 7 runs. the learned boundary wasn't close to the target for most of the runs which is expected given the small sized training set

for training set of size 100, the perceptron converged within 300 or so updates. There was high variability across the runs for number of updates (just eyeballing it) the learned boundary was noticeably comparatively closer than the runs with training set of size 20

for training set of size 1000 all hypothesis were really close to target (just eyeballing it) typically it took anywhere from 200 - 3500 updates for convergence with a good chunk of the 7 runs between 2000 - 3500 updates for convergence

for 10 dim inputs learning algorithm took roughly 3100 updates to converge. I have no way to visually assess how close it was but given the dimension of the space and the thousand training examples I suspect that there was a lot of different ways the boundary could move and still separate the thousand examples

The randomly picked input/updates algorithm was quite a treat drastically decreasing the number of updates needed to separate the data in the vast majority of cases

In general more training examples tracks with higher accuracy in the produced hypothesis when compared with the target and higher dimensional inputs required more updates (from just the 1 run)

randomizing the processing of inputs / updates of misclassified inputs had a drastic impact on number of updates required for convergence.

after analyzing noise graph it seems like the level of noise in the target bounds how good/bad the hypothesis can track the target via the non noisy target. at a noise level of 0.5 it doesn't matter how well/badly the hypothesis tracks the non noisy target the band of performance has width zero at 0.5 error with the target. at higher levels of noise [1.0 - 0.5] we have narrower bands of performance if the target is mostly non noisy, but if the target is mostly noisy (lambda < 0.5) the situation is reversed higher levels of noise leads to wider bands of performance. in both situations we are tracking y by tracking the non noisy part of y.
in general I think the graph is saying that tracking the non noisy part more closely is worthwhile when the target is mostly non noisy but a more laid back approach is better for mostly noisy data. (of course this is speculation)

Problem 1.6
Consider a sample of 10 marbles drawn independently from a bin that holds red and green marbles. The probability of a red marble is mu for mu=0.05, mu=0.5, and mu=0.8 compute the probability of getting no red marbles (v = 0) in the following cases
a) We draw only one such sample. Compute the probability that v = 0

p = np.array([0.05, 0.5, 0.8])
n = 10
k = 0
P(no red marbles) = math.comb(n, k) * ( p ** k) * ((1 - p) ** (n - k)) = array([5.98736939e-01, 9.76562500e-04, 1.02400000e-07]) = approx [0.599, 0.000977, 0.0000001024] in general as the out of sample probability of red increases it becomes less likely that a sample of size 10 doesn't have any of them. (I think actually exponentially unlikely because each chance decrease by a factor of about 1/1000 while the out of sample increase by a factor of 10 and then by a factor a little over 1 so about a 1/10 times increase is leading to roughly the same decrease which I think is decaying exponential behavior)

b) We draw 1000 independent samples. Compute the probability that (at least) one of the samples has v = 0

P(no red marbles in at least 1 of 1000 independence samples) = 1 - P(none of the 1000 samples has no red marbles) = 1 - (1 - P(no red marbles)) ** 1000 = 1 - (1 - array([5.98736939e-01, 9.76562500e-04, 1.02400000e-07])) ** 1000 = array([1.00000000e+00, 6.23576202e-01, 1.02394763e-04]) = approx [1, 0.624, 0.000102] so in general although events become more unlikely the more extreme they are (from part a) over the long run you will likely encounter one and the more extreme the event the longer you will have to run to find one.

c) Repeat (b) for 1000000 independent samples
= array([1.        , 1.        , 0.09733159]) look too hard for something and you'll find it. stop trying to find a pattern where none exist. keep a bolo (be on the lookout) out on the human nature of seeking to tie things up in a nice "cohesive" story bow.

Problem 1.7
A sample of heads and tails is created by tossing a coin a  number of times independently. Assume we have a number of coins that generate different samples independently. For a given coin, let the probability of heads (probability of error) be mu. The probability of obtaining k heads in N tosses of this coin is given by the binomial distribution:

def P(k, N, mu) = math.comb(N, k) * (mu ** k) * (1 - mu) ** (N - k)
Remember the training error v is k / N

a) Assume the sample size (N) is 10. if all the coins have mu=0.5 compute the probability that at least one coin will have v = 0 for the case of 1 coin,
1000 coins, 1000000 coins. Repeat for u=0.8

P(v = 0 in at least 1 of num_coins independence coin flips) = 1 - P(v = 0 for none of the num_coins coin flips) = 1 - (1 - P(v = 0)) ** 1000

= [array([9.765625e-04, 1.024000e-07]), array([6.23576202e-01, 1.02394763e-04]), array([1.        , 0.09733159])]
= approx [[0.000977, 0.000000102], [0.624, 0.0001024], [1, 0.0973]]
I think this is saying it takes a proportional exponential number of tries to be "some what" guaranteed to see extreme events in sample

b) For the case N=6 and 2 coins with mu=0.5 for both coins, plot the probability
P(math.max([math.abs(v[i] - mu) for i in range(2)]) > e)

P(math.max([math.abs(v[i] - mu) for i in range(2)]) > e)
= 1 - P(math.max([math.abs(v[i] - mu) for i in range(2)]) <= e)
let E_0 = math.max([math.abs(v[i] - mu) for i in range(2)]) <= e
then E_1 = math.abs(v[1] - mu) <= e and math.abs(v[2] - mu) <= e is an equivalent event
since v[1] and v[2] are independent 
P(E_1) = P(math.abs(v[1] - mu) <= e) * P(math.abs(v[2] - mu) <= e)

so P(math.max([math.abs(v[i] - mu) for i in range(2)]) > e)
= 1 - P(math.abs(v[1] - mu) <= e) * P(math.abs(v[2] - mu) <= e)

Problem 1.8 The Hoeffding Inequality is one form of the law of large numbers, One of the simplest forms of that law is the Chebyshev Inequality, which you will prove here
a) If t is a non negative random variable, prove that for any alpha > 0, P[t >= alpha] <= E(t) / alpha

(let alpha = beta * E(t) then we are proving P[t >= beta * E(t)] <= 1 / beta. fixing E(t) I think this is saying
the tail mass starting at some multiple of E(t) can be no greater than the reciprocal
of that multiple if E(t) is to remain fixed. this is an extreme bound. only knowing
the expected value is enough to "fix" a portion of the distribution because the expected
value acts as an anchor?)


def I(t, a):
    if t < a:
        return 0
    else:
        return 1

a * I(t, a) <= t since I indicates when t >= a
E(a * I(t, a)) <= E(t) since both random variables are non negative expected value preserves
Inequality
a * E(I(t, a)) <= E(t) - linearity
= a * (0 * P(t < a) + 1 * P(t >= a))
so a * P(t >= a) <= E(t) which gives result 

b) if u is any random variable with mean
mu and variance sigma ** 2, prove that for any
alpha > 0 P[(u - mu) ** 2 >= alpha] <= sigma ** 2 / alpha

E((u - mu) ** 2) = sigma ** 2 by definition
so this is just a restatement of a. since random variable
(u - mu) ** 2 is nonnegative conditions are met.
(I think this is saying examples of really high square
deviation from the mean must become increasingly unlikely the further away
from the mean variance that example is if the mean variance
is actually the mean variance)

c) if u[1]...u[N] are iid random variables, each with
mean mu and variance sigma ** 2 and
v = np.sum[u] / N prove that for any alpha > 0

P[(v - mu) ** 2 >= alpha] <= sigma ** 2 / (N * alpha)

E(v) = E(np.sum[u] / N) = (1 / N) * E(np.sum[u]) = (1 / N) * N * mu = mu by linearity and iid
E((v - mu) ** 2) = sigma(v) ** 2 by definition
need to show sigma(v) ** 2 = sigma ** 2 / N

V(v) = V(np.sum[u] / N) = (1 / N ** 2) * (N * sigma ** 2) proved by myself a long time ago 
= sigma ** 2 / N which gives the result.after applying b)
(I think this is saying that square deviation of sample average from out of sample average by more
than some threshold decays in proportion to (1 / N)) if alpha = beta * sigma ** 2
then we have P[(v - mu) ** 2 >= beta * sigma ** 2] <= 1 / (N * beta)
which says that no more than 1 / (N * beta) "percent" of examples in the distribution of square deviations can have
have values of at least beta * sigma square deviation without moving the average square deviation (variance)

so the story starts with some random variable. (random example) we showed that given the expected value (mean)
no more than 1 / k examples from the distribution can be k times bigger than the mean, hard
(meaning this is a cold stop) this makes sense because if we 
divide the 1 / k examples by k each leaving the (1 / k) examples with with mean value
then the leftover value will be (k - 1) * mean * (1 / k) and distributing this to
the remaining 1 - (1 / k) = (k - 1) / k examples evenly (k - 1) * ( 1 / k) * mean / (k - 1) / k
= mean (markov was indeed a communist I suspect). as if this wasn't interesting enough we shift
the context to the distribution of squared deviations (from the mean) of a random variable (random example) and have a look at that.
it turns out we saw that the same story applies: given the mean square deviation,
 no more than (1 / k) examples of the distribution of square deviations can exceed the mean square deviation by k
times. now this all sounds very interesting but the point is kind of theoretical since having access to the mean of
all the examples is somewhat tricky most of the time. If we have it, we can make bold statements with certainty
if we don't we can only speculate under the scenario that we did have it. the next context attempts to address this
dilemma. If we could just get a good estimate the mean of all the examples we may be in business. We considered
the average of iid random variables. the "expected" value of this random variable is the mean of all the examples
and the expected square deviation (from the mean of all examples) of the average of iid random variables is 0.
given these properties it sounds like the best candidate we have (lol) now
instead of the distribution of a random example, or the distribution of the square deviation (from the mean) of some 
random example, the context is the distribution of averages of N examples from some underlying process.
as with all great abstractions markovs Inequality encapsulates this situation as well.
no more than 1 / k "proportion" of examples of the square deviation of the sample average from the mean (of all examples) 
can exceed the mean square deviation of sample averages (sigma(v) ** 2 == sigma ** 2 / N) by k or more times. 
if we let alpha = k * sigma ** 2 / N then we have  P[(v - mu) ** 2 >= alpha] <= sigma ** 2 / (N * alpha)
= P[(v - mu) ** 2 >= k * (sigma ** 2 / N)] <= 1 / k. so as N increases, the range for the proportion of 
examples (1 / k) exceeding the mean square deviation of the sample average from the mean (of all examples) must
grow wider and wider (tails get diluted of probability mass? tails grow less dense. lol).   

(in the previous problem the approach to show the result was just
to show that the expected value of the transformed variable (that
was the quantity of interest in our prospective inequality)
(square deviation of one random example and square deviation of the average
of N random examples was the same as alpha * RHS of the prospective inequality and
to show that it was just a restatement of the original inequality proven
first which used the fact that if t >= alpha then a <= t lol since
it is tautologically true there was no ambiguity)

(In the below problems we make the transformation from t to e ** (s * t)
(but this is not the quantity of interest in the prospective inequality
after querying "gemini ai" I get the sense that exponential transformations
are useful for examining the tail of a distribution. I believe that transformations
like this is general are used to get a "different" view of the distribution but
I don't have the expertise to say if the "distortion" is so powerful that there
won't be a way to carry the insights back into the original "view")
and we need know if e ** (s * alpha) <= e ** (s * t) when t >= alpha
therefore we need use the fact that e ** (s * t) is monotonically
increasing in t in order to make the inequality. It's easy to skip
over the step if you have a lot of experience with e but the book
makes the point that you should make the point lol which you should
because knowing nothing is the best way to make sure you don't Assume
anything you shouldn't in ml I think)

Problem 1.9)
In this problem, we derive a form of the law of large numbers
that has an exponential bound, called the Chernoff bound.
We focus on the simple case of flipping a fair coin, and
use an approach similar to Problem 1.8

a) Let t be a finite random variable, alpha
be a positive constant, and s be a 
positive parameter. If T(s) = E[t](e ** (s * t)),
prove that P[t >= alpha] <= e ** -(s * a) * T(s)
[Hint e ** st is monotonically increasing in t]



def I(t, a):
    if t < a:
        return 0
    else:
        return 1
let e ** (s * t) be a transformation of t, where s and t are defined as above
since t is a finite random variable e ** (s * t) is also a finite random variable,
the is non negative (increasing in t) therefore it is a candidate for markov's
inequality. 
(e ** (s * a)) * I(t, a) <= e ** (s * t) since I indicates
when t >= a and e grows like a bad weed

=> e ** (s * a) * P(t >= a) <= E[t](e ** (s * t)) by linearity, definition, and
preservation of inequality for non negative random variables
=> e ** (s * a) * P(t >= a) <= T(s) which gives the result


def I(v, a):
    if v < a:
        return 0
    else:
        return 1


b) Let u[1], ..., u[N] be iid random variables, and let
v = np.sum[u] / N. if U(s) = E_u[n](e ** s * u[n]) (for any n) prove that

P[v >= a] <= (e ** -(s * a) * U(s)) ** N

let e ** (s * v) = e ** (s * (np.sum[u] / N)) 
= (e ** (s * (u[1] + ... + u[N]))) ** (1 / N) by algebra
= ((e ** (s * u[1])) * ... * (e ** (s * u[N]))) ** (1 / N) (by algebra)

(e ** (s * a)) * I(v, a) <= e ** (s * v) since I indicates when v >= a and e is
monotonically increasing in v

=> (e ** (s * a)) * I(v, a) <= ((e ** (s * u[1])) * ... * (e ** (s * u[N]))) ** (1 / N)
=> ((e ** (s * a)) ** N) * I(v, a) <= ((e ** (s * u[1])) * ... * (e ** (s * u[N]))) by algebra

so E[((e ** (s * a)) ** N) * I(v, a)] <= E[((e ** (s * u[1])) * ... * (e ** (s * u[N])))]
=>

((e ** (s * a)) ** N) * P[v >= a] <= U(s) ** N by independence (N times) and iid (same U(s)) which gives the result

(in essence, think we found that we can substitute a bound for the tail probability by mapping v to the new space finding
the bound for the tail probability in that space (expected value / threshold value) and using that in the original space
will need to think more on this)

c) Suppose P[u[n] = 0] = P[u[n] = 1] = 1 / 2
(fair coin) Evaluate U(s) as a function of s,
and minimize e ** (-sa) * U(s) with
respect to s for a fixed a, 0 < a < 1.

U(s) = E_u[n](e ** s * u[n]) = (e ** (s * 0)) * (1 / 2) + (e ** (s * 1)) * (1 / 2) = (1 + (e ** s)) / 2
D[(e ** -(s * a)) * (1 + (e ** s)) / 2] = -a * (e ** -(s * a)) * (1 + (e ** s)) / 2 + (e ** -(s * a)) * (e ** s) / 2 = 0
=> -a * (e ** -(s * a)) / 2 + -a * (e ** -(s * a)) * (e ** s) / 2 + (e ** -(s * a)) * (e ** s) / 2 = 0
=> e ** -(s * a) * ((-a / 2) + (-a * (e ** s)) / 2 + (e ** s) / 2) = 0
=> (-a / 2) + (-a * (e ** s)) / 2 + (e ** s) / 2 = 0
=> -a + -a * (e ** s) + (e ** s) = 0
=> -a + (e ** s) * (1 - a) = 0
=> (e ** s) * (1 - a) = a
=> e ** s = a / (1 - a)
=> s = ln(a / (1 - a))

so after averaging out the uncertainty in the bound term we get the smallest bound when the parameter value s = ln(a / (1 - a))
I think the whole purpose of "this" was to get a "nice" bound. (what I would like to do is say something in the context of machine learning so
hopefully when I'm less fried I will be able to do so)

d) Conclude in (c) that, for 0 < epsilon < 1 / 2
P[v >= E[u] + epsilon] <= 2 ** (beta * N)

where beta = 1 + ((1 / 2) + epsilon) * log[2]((1 / 2) + epsilon) + ((1 / 2) - epsilon) * log[2]((1 / 2) - epsilon)
and E[u] = 1 / 2 (expected value of average of iid variables of size N is out of sample mean 1 / 2 for fair coins)
show that beta > 0, hence the bound is exponentially decreasing in N

(I will try evaluating (e ** -(s * a) * U(s)) ** N at the minimum for the d situation
s = ln(a / (1 - a)) and U(s) = U(s) = E_u[n](e ** s * u[n]) = (1 + (e ** s)) / 2 )
(tried doing it with the "code language" but it's looking crazy so will write it out and save
photo in working directory later)
 
 Exercise 2.6
 A data set has 600 examples. To properly test the performance of the final hypothesis, you set aside a randomly selected subset of 200 examples which are never used in the training phase; these form a test set. You use a learning model with 1, 000 hypothesis and slect the final hypothesis g based on the 400 training examples. We wish to estimate E_out(g). We have access to two estimates: E_in(g), the in sample error on the 400 training Examples; and E_test(g), the test error on the 200 test examples that were set aside.
 a) Using a 5% error tolerance (delta = 0.05), which estimate has the higher 'error bar'?
 The E_in(g) has the higher error bar (0.12 vs 0.1)

 b) Is there any reason why you shouldn't reserve even more examples for testing?
 you need enough examples to properly 'tune' your model.
 depending on the line of analysis used, you may be able to get a tighter bound on the sample size needed to get the required performance, thereby freeing up resources to use for test purposes. In general at some point (very quickly in test set size I think) there will be diminishing returns in increasing number of examples used in testing.

 problem 2.2 Show that for the learning model of positive rectangles (aligned horizontally or vertically), m_h(4) = 2 ** 4 and m_h(5) < 2 ** 5. Hence give a bound for m_h(N)

by a previous problem along any given axis, I can shatter at most 2 points. so consider the corresponding dichotomies
of any given two points that I can shatter along one axis. each dichotomy fixes two parameters of the model (either none, one, the other,
or both bits corresponding to the shattered two points are set to 1) because the "interval" along that axis either captures none,
one, the other, or both points along one axis classifying them as positive. So there are only two parameters left to vary along the other axis.
fixing two parameters constrain the points that have the ability to be shattered along the other axis because their first coordinate must lie
in the interval of the corresponding dichotomy, so two bits are used and We need to know how many more bits can we use with the two parameters 
left. any shattering along the other axis that produces the same dichotomy as the first axis contributes nothing to the number of dichotomies.
therefore for any dichotomy we eliminate the 2 bits corresponding to the shattered points. for each dichotomy with the two parameters set
it is optimal that every other available point be within the range of the first coordinates of the already shattered 2 points
(so that the second 2 parameters can apply their full "power" (are free to solely shatter
different points along the other axis) against one coordinate) along the first axis so that they can
potentially be captured by the other two parameters. if there are four points in a square pattern the two of them can be shattered along one axis along
the corresponding coordinate. the other two can be shattered along the other axis along that corresponding coordinate.
to show that 5 points cannot be shattered by the positive rectangle model we consider the rectangles formed by every choice of 2 points from
4 points. where the 2 points form a diagonal of a rectangle. if a point lies between any of these rectangles 
the label for this point will be determined by the two points. all of these rectangles corresponds to a dichotomy in the set of
2 ** 4 dichotomies that can be formed with 4 points. We would like to argue that no extension of these particular 
dichotomies are possible independent of the dichotomy. That is for any particular choice of the fifth point in the
plane we will not be free to choose both -1 and +1 labels independent of these particular dichotomies. The way to see this is to see that there will be at least
2 points above or below the fifth point in either the x or y direction if the fifth point isn't within the rectangle formed by any
2 points (where its label will be determined by those 2 points to be +1). In this situation we can form a rectangle that will enclose one of the two points above
or below the fifth point above, or below the x or y axis so that there is still a constraint of the dichotomy involving those three points
so d_vc = 4 and m_h(N) <= N ** d_vc + 1

for hw3p7, I think for the two positive intervals learning model
we have a maximum of 4 from N + 1 segments we can choose from for N examples
(note choosing a segment is tantamount to generating a unique dichotomy)
for these dichotomies from 4 different segments that are non overlapping in points we get math.comb(N + 1, 4) + 1 for the completely overlapping parts we have math.comb(N + 1, 2) + 1 and we subtract 1 for double counting the dichotomy that doesn't classify any of the points as + 1 (think the general strategy is count completely non overlapping and completely overlapping)

for hw3p9: convex sets - triangle: the restriction is for any given triangle in the real plane, the classification for a point is restricted by whether it lies within any set consisting of the line between any 2 points on the triangle border or not. a triangle consist of 3 segments that are not independent from each other but I think each segment can shatter 2 points (so there is an arrangement of points that allows us to move a given triangle in such a way as to get all 4 possible configurations along each side?) + 1 somehow (idk)

exercise 3.3
Considerthe hat matrix H = X(X^[T]X)^[-1]X^[T], where X is an N by d + 1 matrix, and X^[T]X is invertible


(I currently do not have the mathematical proficiency to easily explain
this symbolically at the moment but the intuition is strong so where I can,
that intuition will be used to make sure that at least it is plausible)

a) Show that H is symmetric (they didn't give definition but fortunately I've look a little bit into this already a matrix A can be shown to be symmetric if A = A^[T])

(X(X^[T]X)^[-1]X^[T])^[T]
= X(X(X^[T]X)^[-1])^[T] because (AB)^[T]_(_, j) = (AB)_(j, _) = A_(j, _)B => (AB)^[T]
= [A_(1, _)B...A_(N, _)B] (to get a the i'th column take the linear combo of the rows of b using the ith row of A as weights. this is the same as if
the columns of B were the rows, B^[T], and the columns of A were the rows, A^[T], giving B^[T]A^[T]) so
X(X(X^[T]X)^[-1])^[T]
= X(X^[T]X)^[-1]^[T]X^[T]
= X(X^[T]X)^[-1]X^[T]
so H^[T] = H
b) show that H^[k] = H for any positive integer K (I know this to be true because H is a projection matrix which is idempotent)

c) if I is the identity matrix of size N, show that (I - H)^[K] = I - H for any positive integer k. I - H is also a projection matrix onto the
nullspace of H (I - H grabs the components of v that can be projected onto the columns of H and subtracts them out of v so we are left with a vector that when multiplied by H will be zero) which is also idempotent

d) show that trace(H) = d + 1, where the trace is the sum of diagonal elements [Hint: trace(AB) = trace(BA)]

(assuming trace is also the sum of eigenvalues, I think to show this result we just need some plausible explaination (the best I can do) that the eigenvalues are 1 and the rank is d + 1)
since this is a projection matrix the eigenvalues are either 1 or 0. This can also be seen because
nullspace(H - lamda * I) = nullspace(H^[k] - lamda * I) 

=> the solutions set of v to (H - lambda * I)v = 0 are the same as the solutions to (H^[k] - lamda * I)v = 0
if v is an eigenvector of H then we must have

lambda^[k] * v - lamda * v = 0
which has solutions lambda = 0, 1

ok next part is H is composed of an invertible part that has d + 1 rows and columns, now although it sits in the center of the expression I'm going to go out on a limb and say the rank of the entire thing (H) is at least d + 1, because of the invertible part with that size and I'm going to go out on a limb and say the rest of the expression just creates some linear combination of those vectors in the invertible part. Intuitively I don't think you can create more linearly independent vectors from a linear combination of less than them so at most its d + 1
combining the 2 intuitive arguments gives the result.

for problem 3.8 I could do a calculation (I have some* facility with this) or I could use intuition (very important to develop good intuition and use it when you can to avoid the arduous use of clock cycles lol) to note that based on experience and previous simplified case average/mean/expected value minimizes square deviation so the hypothesis that minimizes E_out[(h(x) - y) ** 2] will be 
h*(x) = E[Y | x]
ok after looking at a solution although I'm still sure I could come up with the answer analytically I understand that I haven't done any real math in decades. fortunately intuition comes to the rescue. using h*(x) as the target, the deviation for y as a noise term that is data set dependent, we can treat epsilon (the noise term) as a random variable. I know from past experience that the sum of deviations from the mean is 0 in sample (I think) I not too hard to extend that belief to the case where we partition the data calculate the deviation between mean_y and y in partition and say the average of all of them will also be 0. also note for n y's
(y1 - sum(y_i, 1 to n) / n) + (y2 - sum(y_i, 1 to n) / n) + ... + (y_n - 1 - sum(y_i, 1 to n) / n) + (yn - sum(y_i, 1 to n) / n)

= sum(y_i, 1 to n) - n * sum(y_i, 1 to n) / n = 0






