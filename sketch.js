/*
There are 4 essential things in this program :
-Dataset
-Loss function
-Optimizer
-Prediction
*/

//Orginal video by Daniel Shiffman : https://www.youtube.com/watch?v=tIXDik5SGsI

/* Things to add

- A scalable scalar option
- A way to change the optimizer real time
- Comment stuff


*/
let algorithm = 0;

let ys; 

let x_vals = [];
let y_vals = [];

let a, b, c, d

//The higher the learning rate, the faster it will learn however it could cause overfitting
let learningRate = 0.5;

// https://js.tensorflow.org/api/latest/#Training-Optimizers
const optimizerAdam = tf.train.sgd(learningRate)

function setup() {
    createCanvas(600, 600)
    background(0)

    //Tensorflow Variables
    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    d = tf.variable(tf.scalar(random(-1, 1)));


}

function loss(pred, labels) {
    //returns mean
    return pred.sub(labels).square().mean();
}

function mousePressed() {

    //Normalises the values
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);

    x_vals.push(x)
    y_vals.push(y)
}

function predict(x) {
    const xs = tf.tensor1d(x); //The values inputed into the function
    // y = ax^2 + bx + c : This is mathmatically equilivent
    //This is Tensorflow JS in action
    switch(algorithm){
        case 0 : //Linear Regression = y = ax + b
            ys = xs.mul(a).add(b);
        case 1 : //Polynomial Regression with one curve, Ax^2 + bx + c
            ys = xs.square().mul(a).add(xs.mul(b)).add(c);
            break;
        case 2 : //y = Ax^3 + bx^2 + cx + d, Note : The power needs to be a tensor, not an ordinary number
            ys = xs.pow(tf.scalar(3)).mul(a).add(xs.square().mul(b)).add(xs.mul(c)).add(d);
            break;
    }
        
    
    
    // y = Ax^3 + bx^2 + cx + d
    //Note : The power needs to be a tensor, not an ordinary number
    //const ys = xs.pow(tf.scalar(3)).mul(a).add(xs.square().mul(b)).add(xs.mul(c)).add(d);
    return ys;
}



function draw() {

    background(0);
    tf.tidy(() => {
        if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals)
            optimizerAdam.minimize(() => loss(predict(x_vals), ys))
        }
    })


    stroke(255);
    strokeWeight(10);
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], -1, 1, 0, width)
        let py = map(y_vals[i], -1, 1, height, 0)
        point(px, py)
    }

    const curveX = [];

    for (let x = -1; x <= 1.01; x += 0.05) {
        curveX.push(x);
    }
    if(algorithm != 0)
    {
    let xs = [-1, 1];
    const ys = predict(curveX)
    let curveY = ys.dataSync(); //Animation slowdown 


    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);

    for (let i = 0; i < curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width);
        let y = map(curveY[i], 1, -1, 0, height);
        vertex(x, y);
    }
    
    endShape();
    ys.dispose();
    }
    else
    {
        tf.tidy(() => {
        let xs = [0,1];
        const ys = predict(xs)
        let liney= ys.dataSync();

        let x1 = map(xs[0],0,1,0,width)
        let x2 = map(xs[1],0,1,0,width)


        let y1 = map(liney[0],0,1,height,0)
        let y2 = map(liney[1],0,1,height,0)

        line(x1,y1,x2,y2)
        ys.dispose();
        })
    }

    noStroke();
    fill(255,255,255);
    //GUI
    text("learning rate = " + learningRate, 10,20)
}

function keyPressed(){
    
    //Allows the mantipulation of Learning Rate
    switch(key)
        {
            case 'a':
                learningRate +=0.01;
                break;
            case 'd':
                learningRate -=0.01;
                break;
            case 't':
                algorithm++;
                break;
            case 'g':
                algorithm--;
                break;
        }

    console.log(algorithm);
}
