import React from 'react';

// terminal emulator
function DisplayTrainingProgress() {
  const [display, setDisplay] = React.useState(null)

  async function getUpdate() {
    const response = await window.fetch("http://localhost:5000/get_updates", {
      method: "post",
      body: JSON.stringify({}),
    });
    const result = await response.json();
    if (result.nextLine.length > 0) {
      setDisplay(result.nextLine)
    }
  }

  React.useEffect(() => {
      const intervalHandle = setInterval(getUpdate, 1000);
      return () => {
          clearInterval(intervalHandle);
      };
  });

  return (
      <pre style={{
        border: '1px solid black',
        padding: 10,
        margin: 10,
        color: '#494d52',
      }}>
          {display}
      </pre>
  );
}

function App() {
  const [numBatches, setNumBatches] = React.useState('0')
  const [lr, setLr] = React.useState('0')
  const [stackedFfjords, setStackedFfjords] = React.useState('0')
  const [numLayers, setNumLayers] = React.useState('0')
  const [numNodes, setNumNodes] = React.useState('0')
  const [numParams, setNumParams] = React.useState('0')
  const [fixeda, setFixeda] = React.useState('0')
  const [fixedx, setFixedx] = React.useState('0')
  const [fixedy, setFixedy] = React.useState('0')
  const [nPoints, setNPoints] = React.useState('0')

  const [showDisplay, setShowDisplay] = React.useState(false)

  const [datasetImg, setDatasetImg] = React.useState(null)
  const [sampleImg, setSampleImg] = React.useState(null)
  const [fixedaImg, setFixedaImg] = React.useState(null)
  const [fixedxyImg, setFixedxyImg] = React.useState(null)
  const [calibImg, setCalibImg] = React.useState(null)

  
  return (
    <div>
      <div style={{ margin: 20 }}>
        <h1>toy flow demo</h1>
      </div>

      <div style={{ margin: 20 }}>
        <a href="https://arxiv.org/abs/1912.02762">Normalizing flows</a> are a class of generative models known for their expressive power and conceptual simplicity.
        Let us see how a normalizing flow works by examining a toy example.
        We are going to train a <a href="https://arxiv.org/abs/1810.01367">FFJORD</a> to learn a conditional probability density, which is generated from two overlapping 2D Gaussians. 
        Then we are going to visualize what the model has learned, and perform maximum-likelihood inferences with the model.
      </div>

      <div style={{ margin: 20}}>
        <h2>generate dataset</h2>
        The toy problem we are considering is this: 
        There are a bunch of points (x,y) sampled from 2-dimensional Gaussian distributions centered at (a,a).
        We want to train a model to learn the conditional probability distribution, p(x,y|a).
        To do this, let us first generate some training data.
        We first sample a uniformly from the interval [0,1], and then sample one point (x,y) from the 2D Gaussian centered at (a,a), and add the tuple (x,y,a) to the training dataset.
        You can choose how many data points you want to generate, and see a plot of the data points generated here. <br/>
        <br/>
        dataset size &nbsp; <input
          type='text'
          value={numBatches}
          onChange={(event) => {
            setNumBatches(event.target.value);
          }}
        /> * 256 (batch size) = {numBatches * 256}<br/>
        <span style={{fontSize: 14}}>(recommended: 100; best: 1600)</span> <br/><br/>
        <button onClick={async () => {
          const response = await window.fetch('http://127.0.0.1:5000/generate_dataset', {
            method: 'POST',
            body: JSON.stringify({
            num_batches: numBatches,
            }),
          });
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setDatasetImg(json.pngData);
        }}>
          generate
        </button><br/>
      </div>

      {
        datasetImg !== null &&
        <img
          style={{
            width: 400,
          }}
          src={datasetImg}
        />
      }

      <div style={{ margin: 20}}>
        <h2>
          model and training
        </h2>
        Now, pick the model parameters. 
        We are using a series of FFJORDs, and you can specify the number of FFJORDs, the number of hidden layers in each FFJORD, the number of nodes per layer, as well as the initial learning rate. <br/><br/>
        <table>
          <tbody>
            <tr>
              <td>learning rate</td>
              <td><input
                    type='text'
                    value={lr}
                    onChange={(event) => {
                       setLr(event.target.value);
                   }}
                /></td>
            </tr>
            <tr>
              <td>stacked ffjords</td>
              <td><input
                    type='text'
                    value={stackedFfjords}
                    onChange={(event) => {
                       setStackedFfjords(event.target.value);
                   }}
                /></td>
            </tr>
            <tr>
              <td>hidden layers</td>
              <td><input
                    type='text'
                    value={numLayers}
                    onChange={(event) => {
                       setNumLayers(event.target.value);
                   }}
                /></td>
            </tr>
            <tr>
              <td>nodes per layer</td>
              <td><input
                    type='text'
                    value={numNodes}
                    onChange={(event) => {
                       setNumNodes(event.target.value);
                   }}
                /> * 2 (number of output dimensions) = {numNodes*2}
              </td>
            </tr>
          </tbody>
        </table>
        <span style={{fontSize: 14}}>(recommended: 0.001, 3, 3, 4; best: 0.0005, 5, 3, 12)</span>
      </div>
      <div style={{ margin: 20}}>
      Before we start the training process, let's see how many parameters are in the model you specified. <br/><br/>
        <button onClick={async () => {
          const response = await window.fetch('http://127.0.0.1:5000/calculate_num_params', {
            method: 'POST',
            body: JSON.stringify({
            lr: lr,
            stacked_ffjords: stackedFfjords,
            num_layers: numLayers,
            num_nodes: numNodes,
            }),
          });
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setNumParams(json.numParams);
        }}>
          calculate number of parameters
        </button>
        &nbsp;&nbsp;
        {numParams} 
        <br/><br/>
        Now, let's train the model! 
        The training process uses Keras, and uses a train/validation split of 90/10. 
        It monitors the validation loss, and has callbacks EarlyStopping and ReducedLROnPlateau.
        <br/>
        For the "recommended" values, expect the first epoch to take about a minute, and subsequent epochs to take about 5 seconds.
        For the "best" values, expect each epoch to take on the order of 5 minutes. <br/><br/>
        <button onClick={async () => {
          const response = await window.fetch('http://127.0.0.1:5000/train_model', {
            method: 'POST',
            body: JSON.stringify({
            lr: lr,
            stacked_ffjords: stackedFfjords,
            num_layers: numLayers,
            num_nodes: numNodes,
            }),
          });
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setShowDisplay(true);
        }}>
          train!
        </button>
        <br/><br/>
        { showDisplay && <DisplayTrainingProgress/> }
        Let's take some samples from the model and see how it learned.
        Let's take the same number of samples from the model as the dataset size specified above.
       <br/><br/>
        <button onClick={async () => {
          const response = await window.fetch('http://127.0.0.1:5000/plot_samples', {
            method: 'POST',
            body: JSON.stringify({
            num_batches: numBatches,
            stacked_ffjords: stackedFfjords,
            num_nodes: numNodes,
            num_layers: numLayers,
            }),
          });
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setSampleImg(json.pngData);
        }}>
          sample and plot
        </button>
      </div>

      {
        sampleImg !== null &&
        <div style={{ margin: 20}}>
          <img
          style={{
            width: 800,
          }}
          src={sampleImg}
        /><br/>
        Each color contains the same number of points as the dataset size you specified above.
        The <span style={{color: "#fc2121"}}>red</span> points are sampled with the same conditional values as the ones in the training set.
        The <span style={{color: "#2128fc"}}>blue</span> points are sampled with a conditional value of 0, and the <span style={{color: "#008c02"}}>green</span> points are sampled with a conditional value of 1.
        Do the <span style={{color: "#2128fc"}}>blue</span> and <span style={{color: "#008c02"}}>green</span> points look like 2D Gaussians centered at (0,0) and (1,1)?
        </div>
      }

      <div style={{ margin: 20}}>
        <h2>
          maximum-likelihood inference
        </h2>
        We can now use the flow to perform inference.
        For example, we can answer questions such as: given a point (x,y), what is the mostly likely conditional value?
        i.e. What is a such that it is most likely that (x,y) had been sampled from the Gaussian centered at (a,a)?
        <br/>
        For the this toy problem, note that there is a simple analytical answer: (x+y)/2.
        <br/><br/>
        One of the most prominent benefits of a normalizing flow is that it allows us to simultaneously draw samples and evaluate densities.
        Let us first take a look at the density that our flow has learned. 
        For example, we can check both p(x,y|a) as a function of x (or y) for a fixed a, and p(x,y|a) as a function of a for a fixed (x,y).
        
        <h3>
          p(x,y|a) for fixed a
        </h3>
        a &nbsp; <input
          type='text'
          value={fixeda}
          onChange={(event) => {
            setFixeda(event.target.value);
          }}
        /> (between 0 and 1) <br/><br/>
        <button onClick={async () => {
          const response = await window.fetch('http://127.0.0.1:5000/plot_fixed_a', {
            method: 'POST',
            body: JSON.stringify({
            num_batches: numBatches,
            stacked_ffjords: stackedFfjords,
            num_nodes: numNodes,
            num_layers: numLayers,
            fixed_a: fixeda,
            }),
          });
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setFixedaImg(json.pngData);
        }}>
          plot
        </button> &nbsp; (this might take ~5 seconds)<br/>
        {
          fixedaImg !== null &&
          <div>
            <img
              style={{
                width: 500,
              }}
              src={fixedaImg}
            /><br/>
            Here we are plotting the slice along y=x.
          </div>
        }

        <h3>
          p(x,y|a) for fixed (x,y)
        </h3>
        x &nbsp; <input
          type='text'
          value={fixedx}
          onChange={(event) => {
            setFixedx(event.target.value);
          }}
        /> (between 0 and 1) <br/>
        y &nbsp; <input
          type='text'
          value={fixedy}
          onChange={(event) => {
            setFixedy(event.target.value);
          }}
        /> (between 0 and 1) <br/><br/>
        <button onClick={async () => {
          const response = await window.fetch('http://127.0.0.1:5000/plot_fixed_xy', {
            method: 'POST',
            body: JSON.stringify({
            num_batches: numBatches,
            stacked_ffjords: stackedFfjords,
            num_nodes: numNodes,
            num_layers: numLayers,
            fixed_x: fixedx,
            fixed_y: fixedy,
            }),
          });
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setFixedxyImg(json.pngData);
        }}>
          plot
        </button><br/>
        {
          fixedxyImg !== null &&
          <img
            style={{
              width: 500,
            }}
            src={fixedxyImg}
          />
        }

        <h3>
          calibration curve
        </h3>
        With the learned density, we could now produce a calibration curve:
        We generate a bunch of random points, and plot the most likely conditional value a that the model infers for each point against the true value.
        The closer it is to a straight, diagonal line, the more calibrated our model is.
        <br/><br/>
        number of points &nbsp; <input
          type='text'
          value={nPoints}
          onChange={(event) => {
            setNPoints(event.target.value);
          }}
        /> (recommended: 100) <br/><br/>
        <button onClick={async () => {
          const response = await window.fetch('http://127.0.0.1:5000/calibration_curve', {
            method: 'POST',
            body: JSON.stringify({
            num_batches: numBatches,
            stacked_ffjords: stackedFfjords,
            num_nodes: numNodes,
            num_layers: numLayers,
            n_points: nPoints,
            }),
          });
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setCalibImg(json.pngData);
        }}>
          plot
        </button><br/>
        {
          calibImg !== null &&
          <img
            style={{
              width: 500,
            }}
            src={calibImg}
          />
        }
      </div>
      
      <div style={{ margin: 20 }}>
        <h2>physics application: particle reconstruction at the LHC</h2>
        With this toy example, we demonstrated a functioning normalizing flow that is capable of learning a conditional probability distribution, and perform maximum-likelihood inference with it.
        We are now working on using this technique to unify simulation and particle reconstruction at the LHC, similar to some previous work named <a href="https://arxiv.org/abs/2106.05285">CaloFlow</a>.
        Our research code is at <a href="https://github.com/ViniciusMikuni/ToyFlow">this GitHub repo</a>.
        Stay tuned!
      </div>
    </div>
  );
}

export default App;


