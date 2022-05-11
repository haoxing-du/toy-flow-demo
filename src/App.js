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

  const [datasetImg, setDatasetImg] = React.useState(null)
  const [modelImg, setModelImg] = React.useState(null)

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

  return (
    <div>
      <div style={{ margin: 20 }}>
        <h1>toy flow demo</h1>
      </div>

      <div style={{ margin: 20 }}>
        Let us see how a normalizing flow works by examining a toy example.
        We are going to train a FFJORD to learn a conditional probability density, which is generated from two overlapping 2D Gaussians. 
        Then we are going to perform maximum-likelihood inferences with the model, and demonstrate prior-independence.
      </div>

      <div style={{ margin: 20}}>
        <h2>generate dataset</h2>
        First, let's generate the dataset. <br/>
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
          const response = await window.fetch('http://127.0.0.1:5000/visualize_model', {
            method: 'POST',
            body: JSON.stringify({
            lr: lr,
            stacked_ffjords: stackedFfjords,
            num_layers: numLayers,
            num_nodes: numNodes*2
            }),
          });
          const json = await response.json();
          console.log('JSON:', json);
          window.sneakyJson = json;
          setDatasetImg(json.pngData);
        }}>
          visualize model
        </button>
        &nbsp;&nbsp;
        <button>
          train!
        </button>
        <br/><br/>
        Let's take some samples from the model and see how it learned.
      </div>

      <div>
        <h2>
          maximum-likelihood inference
        </h2>

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


