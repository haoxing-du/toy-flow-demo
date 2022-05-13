import React from 'react';

function ListElement(props) {
  return (
    <div style={{
      border: '1px solid black',
      padding: 10,
      margin: 10,
      backgroundColor: '#808080',
      color: 'white',
    }}>
      {props.label}: {props.value}
    </div>
  );
}

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
  const [counter, setCounter] = React.useState(0);
  const [list, setList] = React.useState(
    [
      {
        label: 'hello',
        value: 3,
      },
      {
        label: 'flob',
        value: 5,
      },
    ]
  );
  
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

  const [textBox, setTextBox] = React.useState('0');
  const [requestResult, setRequestResult] = React.useState(null);

  const renderedListElements = [];
  for (const entry of list) {
    renderedListElements.push(
      <ListElement
        label={entry.label}
        value={entry.value}
      />
    );
  }
  const [value, setValue] = React.useState(0);

  
  return (
    <div>
      <div style={{ margin: 20 }}>
        <h1>toy flow demo</h1>
      </div>

      <div style={{ margin: 20 }}>
        Normalizing flows are a class of generative models known for their expressive power and conceptual simplicity.
        Let us see how a normalizing flow works by examining a toy example.
        We are going to train a FFJORD to learn a conditional probability density, which is generated from two overlapping 2D Gaussians. 
        Then we are going to perform maximum-likelihood inferences with the model, and demonstrate prior-independence.
      </div>

      <div style={{ margin: 20}}>
        <h2>generate dataset</h2>
        The toy problem we are considering is this: there is a bunch of points sampled from 2-dimensional Gaussian distributions centered at (a,a).
        We want to train a model to infer the center of the Gaussian that it most probably came from, for each given point.
        To do this, let us first generate some training data.
        We sample a uniformly from the interval [0,1], and then sample one point from the 2D Gaussian centered at (a,a), and add the coordinates of this point together with a to the training dataset.
        You can choose how many data points you want to generate, and see a plot of the data points generated here. <br/>
        <br/>
        Dataset size &nbsp; <input
          type='text'
          value={numBatches}
          onChange={(event) => {
            setNumBatches(event.target.value);
          }}
        /> * 256 (batch size) = {numBatches * 256}<br/>
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
        Now, pick the model parameters: <br/><br/>
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
      </div>
      <div style={{ margin: 20}}>
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
        Let's take some samples from the model and see how it learned. <br/><br/>
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
        <img
          style={{
            width: 800,
          }}
          src={sampleImg}
        />
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
        For example, we can check both p(x|a) as a function of x for a fixed a, and p(x|a) as a function of a for a fixed x.
        
        <h3>
          p(x|a) for fixed a
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
        </button><br/>
        {
          fixedaImg !== null &&
          <img
            style={{
              width: 500,
            }}
            src={fixedaImg}
          />
        }

        <h3>
          p(x|a) for fixed x
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
        <br/>
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




      <br/>
      <br/>
      Counter: {counter}<br/>

      <button onClick={() => {
        setCounter(counter + 1);
      }}>
        Increase value
      </button>

      <div>
        {renderedListElements}
      </div>
      <button onClick={() => {
        setList(
          [...list, {
            label: 'new',
            value: Math.random(),
          }],
        );
      }}
      >
        Create new entry
      </button>

      <div style={{ margin: 20 }}>
        Value: <input
          type='text'
          value={textBox}
          onChange={(event) => {
            setTextBox(event.target.value);
          }}
        /><br/>

        <button onClick={async () => {
          const response = await window.fetch('http://localhost:5000/plot_function', {
            method: 'POST',
            body: JSON.stringify({
              y_func: textBox,
            }),
          });
          //console.log('Response:', await response.text());
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setRequestResult(json.pngData);
        }}>
          Plot function
        </button><br/>

        {/* Result: {requestResult} */}
      </div>

      {
        requestResult !== null &&
        <img
          style={{
            width: 200,
          }}
          src={requestResult}
        />
      }
    </div>
  );
}

export default App;


