Download Link: https://assignmentchef.com/product/solved-b555-project-3-bayesian-glm
<br>
In this programming project we will be working with Generalized Linear Models as covered in class, including Logistic regression, Poisson regression and Ordinal regression. Your goal is to use one generic implementation for the main algorithm that works for multiple observation likelihoods.

<h1>Data</h1>

Data for this assignment is provided in a zip file pp3data.zip on Canvas. Each dataset is given in two files with the data in one and the labels in the other file. We will use the datasets A and usps for classification with logistic regression. We will use the datasets AP for count prediction with Poisson regression. We will use the datasets AO for ordinal prediction with ordinal regression.

The datasets A, AP, AO were artificially generated with labels which are not perfectly matched to any linear predictor yet they are generated to be somewhat predictable. The examples in usps represent 16×16 bitmaps of the characters 3 and 5 and are taken from the well known usps dataset (representing data originally used for zip code classification).

<h1>Implementing our variant of GLM</h1>

In this assignment we will use a Bayesian (or regularized) version of the algorithm with <em>w </em>∼ ), where <em>α </em>= 10, to calculate the MAP solution <em>w<sub>MAP </sub></em>.

<ul>

 <li>As discussed in class logistic regression (and by extension GLM) relies on a free parameter (<em>w</em><sub>0</sub>) to capture an appropriate separating hyperplane. Therefore, you will need to add a feature fixed at one (also known as an intercept) to all datasets in the assignment. To match the test case below please add this as the first column in the data matrix.</li>

 <li>The vector of first derivatives of the log posterior is <em>g </em>= <em><u><sup>∂LogL</sup></u><sub>∂w </sub></em>= <sup>P</sup><em><sub>i </sub>d<sub>i</sub>φ</em>(<em>x<sub>i</sub></em>)−<em>αw </em>= Φ<em><sup>T </sup>d</em>−<em>αw </em>where <em>d </em>is a vector whose elements are <em>d<sub>i</sub></em>.</li>

 <li>The matrix of second derivatives of the log posterior is <em>H </em>= <em>∂w∂w<u><sup>∂LogL</sup></u></em><em><sub>T </sub></em>= −<sup>P</sup><em><sub>i </sub>r<sub>i</sub>φ</em>(<em>x<sub>i</sub></em>)<em>φ</em>(<em>x<sub>i</sub></em>)<em><sup>T </sup></em>− <em>αI </em>= −Φ<em><sup>T </sup>R</em>Φ − <em>αI </em>where <em>R </em>is a matrix with elements <em>r<sub>i </sub></em>on the diagonal.</li>

 <li>The GLM algorithm initializes the weight vector as <em>w </em>= 0 and then repeatedly applies an update with Netwon’s method <em>w </em>← <em>w </em>− <em>H</em><sup>−1</sup><em>g </em>until <em>w </em></li>

 <li>For this assignment we consider that <em>w </em>has converged if . If <em>w </em>has not converged in 100 iterations we stop and output the last <em>w </em>as our solution.</li>

 <li>The final vector, when the algorithm stops is <em>w<sub>MAP </sub></em>. In this assignment we will use <em>w<sub>MAP </sub></em>for prediction (i.e. we will not calculate a predictive distribution).</li>

</ul>

<h1>Likelihood models</h1>

<ul>

 <li>In order to apply the algorithm for any likelihood model and to evaluate its predictions we need to specify 4 items: (1) <em>d<sub>i</sub></em>, (2) <em>r<sub>i</sub></em>, (3) how to compute our prediction <em>t</em>ˆ for test example <em>z</em>, and (4) how to calculate the error when we predict <em>t</em>ˆand the true label is <em>t</em>.</li>

 <li>For the logistic likelihood we have: <em>y<sub>i </sub></em>= <em>σ</em>(<em>w<sup>T </sup>φ</em>(<em>x<sub>i</sub></em>)) and the first derivative term is <em>d<sub>i </sub></em>= <em>t<sub>i</sub></em>−<em>y<sub>i </sub></em>or <em>d </em>= <em>t </em>− <em>y</em>. The second derivative term is <em>r<sub>i </sub></em>= <em>y<sub>i</sub></em>(1 − <em>y<sub>i</sub></em>). For test example <em>z </em>we predict <em>t </em>= 1 iff <em>p</em>(<em>t </em>= 1) = <em>σ</em>(<em>w<sub>MAP</sub><sup>T </sup>φ</em>(<em>z</em>)) ≥ 0<em>.</em> The error is 1 if <em>t</em>ˆ6= <em>t</em>.</li>

 <li>Note that for the logistic model the update formula as developed in class is <em>w<sub>n</sub></em><sub>+1 </sub>← <em>w<sub>n </sub></em>− (−<em>αI </em>− Φ<em><sup>T </sup>R</em>Φ)<sup>−1</sup>[Φ<em><sup>T </sup></em>(<em>t </em>− <em>y</em>) − <em>αw<sub>n</sub></em>]. You might want to start developing your code and testing it with this special case and then generalize it to handle all likelihoods. To help you test your implementation of this algorithm we provide an additional dataset, <em>irlstest</em>, and solution weight vector in <em>irlsw </em>(for <em>α </em>= 0<em>.</em>1). The first entry in <em>irlsw </em>corresponds to <em>w</em><sub>0</sub>.</li>

 <li>For the Poisson likelihood we have: <em>y<sub>i </sub></em>= <em>e</em><sup>(<em>w</em></sup><em><sup>Tφ</sup></em><sup>(<em>x</em></sup><em><sup>i</sup></em><sup>)) </sup>and the first derivative term is <em>d<sub>i </sub></em>= <em>t<sub>i</sub></em>−<em>y<sub>i </sub></em>or <em>d </em>= <em>t</em>−<em>y</em>. The second derivative term is <em>r<sub>i </sub></em>= <em>y<sub>i</sub></em>. For test example <em>z </em>we have <em>p</em>(<em>t</em>) = <em>Poisson</em>(<em>λ</em>) where <em>a </em>= <em>w<sub>MAP</sub><sup>T </sup>φ</em>(<em>z</em>) and <em>λ </em>= <em>e<sup>a</sup></em>. We predict the mode <em>t </em>= b<em>λ</em> For this assignment we will use the absolute error: err = |<em>t</em>ˆ− <em>t</em>|.</li>

 <li>For the ordinal model with <em>K </em>levels we have parameters <em>s </em>and <em>φ</em><sub>0 </sub>= −∞ <em>&lt; φ</em><sub>1 </sub><em>&lt; … &lt; φ<sub>K</sub></em><sub>−1 </sub><em>&lt; φ<sub>K </sub></em>= ∞ where for this assignment we will use <em>K </em>= 5, <em>s </em>= 1 and <em>φ</em><sub>0 </sub>= −∞ <em>&lt; φ</em><sub>1 </sub>= −2 <em>&lt; φ</em><sub>2 </sub>= −1 <em>&lt; φ</em><sub>3 </sub>= 0 <em>&lt; φ</em><sub>4 </sub>= 1 <em>&lt; φ</em><sub>5 </sub>= ∞. The model is somewhat sensitive to the setting of hyperparameters so it is important to use these settings.</li>

</ul>

Here <em>a<sub>i </sub></em>= <em>w<sup>T </sup>φ</em>(<em>x<sub>i</sub></em>) and for <em>j </em>∈ {0<em>,…K</em>} we have <em>y<sub>i,j </sub></em>= <em>σ</em>(<em>s</em>∗(<em>φ<sub>j </sub></em>−<em>a<sub>i</sub></em>)). Using this notation, for example <em>i </em>with label <em>t<sub>i </sub></em>we have <em>d<sub>i </sub></em>= <em>y<sub>i,t</sub></em><em><sub>i </sub></em>+<em>y<sub>i,</sub></em><sub>(<em>t</em></sub><em><sub>i</sub></em>−<sub>1) </sub>−1. For the second derivative we have

<em>r</em><em>i </em>= <em>s</em><sup>2</sup>[<em>y</em><em>i,t<sub>i</sub></em>(1 − <em>y</em><em>i,t<sub>i</sub></em>) + <em>y<sub>i,</sub></em>(<em>t<sub>i</sub></em>−1)(1 − <em>y<sub>i,</sub></em>(<em>t<sub>i</sub></em>−1))].

To predict for test example <em>z </em>we first calculate the <em>y </em>values: <em>a </em>= <em>w<sub>MAP</sub><sup>T </sup>φ</em>(<em>z</em>) and for <em>j </em>∈ {0<em>,…K</em>} we have <em>y<sub>j </sub></em>= <em>σ</em>(<em>s </em>∗ (<em>φ<sub>j </sub></em>− <em>a</em>)). Then for potential labels <em>j </em>∈ {1<em>,…K</em>} we calculate <em>p<sub>j </sub></em>= <em>y<sub>j </sub></em>− <em>y<sub>j</sub></em><sub>−1</sub>. Finally select <em>t</em>ˆ= argmax<em><sub>j </sub>p<sub>j</sub></em>. For this assignment we will use the absolute error, or the number of levels we are off in prediction, that is, err = |<em>t</em>ˆ− <em>t</em>|.

While you could implement these as three separate algorithms, you are expected to provide one implementation of the main optimization which is given access to procedures calculating the 4 items above to make a concrete instance of GLM.

<h1>Evaluating the implementation</h1>

Your task is to implement the GLM algorithm and generate learning curves with error bars (i.e., ±1<em>σ</em>) as follows.

Repeat 30 times

Step 1) Set aside 1/3 of the total data (randomly selected) to use as a test set.

Step 2) Permute the remaining data and record the test set error rate as a function of increasing training set portion (0.1,0.2, …,1 of the total size).

Calculate the mean and standard deviation for each size and plot the result. In addition record the number of iterations and the run time untill convergence in each run and report their averages.

In your submission provide 4 plots, one for each dataset, and corresponding runtime/iterations statistics, and provide a short discussion of the results. Are the learning curves as expected? how does learning time vary across datasets for classification? and across the likelihood models? what are the main costs affecting these (time per iteration, number of iterations)?

<h1>Extra Credit</h1>

Explore some approach for model selection for <em>α </em>in all models and/or for <em>s </em>and <em>φ </em>in the ordinal model and report your results. You may want to generate your own data with known parameters in order to test the success of algorithms in identifying good parameters.