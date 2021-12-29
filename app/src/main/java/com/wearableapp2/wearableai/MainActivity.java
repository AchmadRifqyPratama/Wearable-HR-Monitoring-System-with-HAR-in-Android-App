package com.wearableapp2.wearableai;

import static android.content.ContentValues.TAG;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

public class MainActivity extends AppCompatActivity{

    private String deviceName = null;
    private String deviceAddress;
    public static Handler handler;
    public static BluetoothSocket mmSocket;
    public static ConnectedThread connectedThread;
    public static CreateConnectThread createConnectThread;
    private final static int CONNECTING_STATUS = 1; // used in bluetooth handler to identify message status
    private final static int NEW_SEQUENCE = 2; // used in bluetooth handler to identify new sequence of data receiving
    private final static int R_VALUE = 3; // used in bluetooth handler to identify ppg value of data receiving
    private final static int IR_VALUE = 8; // used in bluetooth handler to identify ppg value of data receiving
    private final static int X_VALUE = 4; // used in bluetooth handler to identify x value of data receiving
    private final static int Y_VALUE = 5; // used in bluetooth handler to identify y value sequence of data receiving
    private final static int Z_VALUE = 6; // used in bluetooth handler to identify z value sequence of data receiving

    int intBytes = 0;
    int i = 0;
    int counter = 0;
    float movingAvrg = 0;
    int k = 0;
    String strtoggleLED = "";
    private static String arduinoMsg = "";
    private static String nilaiBPM = "";
    boolean run = false;

    private final int N_Samples = 100;
    private static List<Float> ax, ay, az;
    private static List<Integer> ppg1, ppg2;

    private static float[][] results;
    private HARClassifier classifier;
    private TextView restingTextView;
    private TextView norestingTextView;

    TextView detakJantung;
    TextView abnormalitas;
    TextView rValue;
    TextView irValue;
    TextView xValue;
    TextView yValue;
    TextView zValue;

    final int fs = 100;
    final int K=8*fs;
    final int DK=1*fs;
    int j=0;
    int fin=0;
    int window=8;
    int step=1;
    int ind;
    float heartbeat=0;
    float z_ppg1[] = new float[100];
    float z_ppg2[] = new float[100];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // UI Initialization
        final Button buttonConnect = findViewById(R.id.buttonConnect);
        final Toolbar toolbar = findViewById(R.id.toolbar);
        final ProgressBar progressBar = findViewById(R.id.progressBar);
        progressBar.setVisibility(View.GONE);
        final Button buttonToggle = findViewById(R.id.buttonToggle);
        buttonToggle.setEnabled(false);

        rValue = findViewById(R.id.rValue);
        irValue = findViewById(R.id.irValue);
        xValue = findViewById(R.id.xValue);
        yValue = findViewById(R.id.yValue);
        zValue = findViewById(R.id.zValue);
        abnormalitas = findViewById(R.id.abnormalitas);
        detakJantung = findViewById(R.id.detakJantung);

        for(int k=0; k<100; k++){
            z_ppg1[i] = 1;
            z_ppg2[i] = 1;
        }

        restingTextView = findViewById(R.id.restingTextView);
        norestingTextView = findViewById(R.id.norestingTextView);

        ax = new ArrayList<>();
        ay = new ArrayList<>();
        az = new ArrayList<>();
        ppg1 = new ArrayList<>();
        ppg2 = new ArrayList<>();

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        try{
            classifier = new HARClassifier(getApplicationContext()); //try/catch?
        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }

        // If a bluetooth device has been selected from SelectDeviceActivity
        deviceName = getIntent().getStringExtra("deviceName");
        if (deviceName != null){
            // Get the device address to make BT Connection
            deviceAddress = getIntent().getStringExtra("deviceAddress");
            // Show progree and connection status
            toolbar.setSubtitle("Connecting to " + deviceName + "...");
            progressBar.setVisibility(View.VISIBLE);
            buttonConnect.setEnabled(false);

            /*
            This is the most important piece of code. When "deviceName" is found
            the code will call a new thread to create a bluetooth connection to the
            selected device (see the thread code below)
             */
            BluetoothAdapter bluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
            createConnectThread = new CreateConnectThread(bluetoothAdapter,deviceAddress);
            createConnectThread.start();
        }

        /*
        Second most important piece of Code. GUI Handler
         */
        handler = new Handler(Looper.getMainLooper()) {
            @Override
            public void handleMessage(Message msg){
                switch (msg.what){
                    case CONNECTING_STATUS:
                        switch(msg.arg1){
                            case 1:
                                toolbar.setSubtitle("Connected to " + deviceName);
                                progressBar.setVisibility(View.GONE);
                                buttonConnect.setEnabled(true);
                                buttonToggle.setEnabled(true);
                                break;
                            case -1:
                                toolbar.setSubtitle("Device fails to connect");
                                progressBar.setVisibility(View.GONE);
                                buttonConnect.setEnabled(true);
                                break;
                        }
                        break;

                    case R_VALUE:
                        arduinoMsg = msg.obj.toString();
                        ppg1.add(Integer.valueOf(arduinoMsg.substring(0,arduinoMsg.length()-1)));
                        rValue.setText("Red Value : " + arduinoMsg);
                        break;

                    case IR_VALUE:
                        arduinoMsg = msg.obj.toString();
                        ppg2.add(Integer.valueOf(arduinoMsg.substring(0,arduinoMsg.length()-1)));
                        irValue.setText("InfraRed Value : " + arduinoMsg);
                        break;

                    case X_VALUE:
                        arduinoMsg = msg.obj.toString();
                        ax.add(Float.valueOf(arduinoMsg.substring(0,arduinoMsg.length()-1)));
                        xValue.setText("X Value : " + arduinoMsg);
                        break;

                    case Y_VALUE:
                        arduinoMsg = msg.obj.toString();
                        ay.add(Float.valueOf(arduinoMsg.substring(0,arduinoMsg.length()-1)));
                        yValue.setText("Y value : " + arduinoMsg);
                        break;

                    case Z_VALUE:
                        arduinoMsg = msg.obj.toString();
                        az.add(Float.valueOf(arduinoMsg.substring(0,arduinoMsg.length()-1)));
                        zValue.setText("Z value : " + arduinoMsg);
                        break;
                }

                HARPrediction();
                HRProcessing();
            }

        };

        // Select Bluetooth Device
        buttonConnect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Move to adapter list
                Intent intent = new Intent(MainActivity.this, SelectDeviceActivity.class);
                startActivity(intent);
            }
        });

        // Button to ON/OFF LED on Arduino Board
        buttonToggle.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String btnState = buttonToggle.getText().toString().toLowerCase();
                switch (btnState) {
                    case "receive":
                        buttonToggle.setText("Stop Receive");
                        // Command to turn on LED on Arduino. Must match with the command in Arduino code
                        strtoggleLED = "1";
                        run = true;
                        connectedThread.write(strtoggleLED);

                        break;
                    case "stop receive":
                        buttonToggle.setText("Receive");
                        // Command to turn off LED on Arduino. Must match with the command in Arduino code
                        strtoggleLED = "0";
                        run = false;
                        connectedThread.write(strtoggleLED);
                        i = 0;
                        break;
                }
            }
        });

    }

    /* ============================ Thread to Create Bluetooth Connection =================================== */
    public static class CreateConnectThread extends Thread {

        public CreateConnectThread(BluetoothAdapter bluetoothAdapter, String address) {
            /*
            Use a temporary object that is later assigned to mmSocket
            because mmSocket is final.
             */
            BluetoothDevice bluetoothDevice = bluetoothAdapter.getRemoteDevice(address);
            BluetoothSocket tmp = null;
            UUID uuid = bluetoothDevice.getUuids()[0].getUuid();

            try {
                /*
                Get a BluetoothSocket to connect with the given BluetoothDevice.
                Due to Android device varieties,the method below may not work fo different devices.
                You should try using other methods i.e. :
                tmp = device.createRfcommSocketToServiceRecord(MY_UUID);
                 */
                tmp = bluetoothDevice.createInsecureRfcommSocketToServiceRecord(uuid);

            } catch (IOException e) {
                Log.e(TAG, "Socket's create() method failed", e);
            }
            mmSocket = tmp;
        }

        public void run() {
            // Cancel discovery because it otherwise slows down the connection.
            BluetoothAdapter bluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
            bluetoothAdapter.cancelDiscovery();
            try {
                // Connect to the remote device through the socket. This call blocks
                // until it succeeds or throws an exception.
                mmSocket.connect();
                Log.e("Status", "Device connected");
                handler.obtainMessage(CONNECTING_STATUS, 1, -1).sendToTarget();
            } catch (IOException connectException) {
                // Unable to connect; close the socket and return.
                try {
                    mmSocket.close();
                    Log.e("Status", "Cannot connect to device");
                    handler.obtainMessage(CONNECTING_STATUS, -1, -1).sendToTarget();
                } catch (IOException closeException) {
                    Log.e(TAG, "Could not close the client socket", closeException);
                }
                return;
            }

            // The connection attempt succeeded. Perform work associated with
            // the connection in a separate thread.
            connectedThread = new ConnectedThread(mmSocket);
            connectedThread.run();
        }

        // Closes the client socket and causes the thread to finish.
        public void cancel() {
            try {
                mmSocket.close();
            } catch (IOException e) {
                Log.e(TAG, "Could not close the client socket", e);
            }
        }
    }

    /* =============================== Thread for Data Transfer =========================================== */
    public static class ConnectedThread extends Thread {
        private final BluetoothSocket mmSocket;
        private final InputStream mmInStream;
        private final OutputStream mmOutStream;
        private int i=0;

        public ConnectedThread(BluetoothSocket socket) {
            mmSocket = socket;
            InputStream tmpIn = null;
            OutputStream tmpOut = null;

            // Get the input and output streams, using temp objects because
            // member streams are final
            try {
                tmpIn = socket.getInputStream();
                tmpOut = socket.getOutputStream();
            } catch (IOException e) { }

            mmInStream = tmpIn;
            mmOutStream = tmpOut;
        }

        public void run() {
            byte[] buffer = new byte[1024];  // buffer store for the stream
            int bytes = 0; // bytes returned from read()
            // Keep listening to the InputStream until an exception occurs
            while (true) {
                try {
                    /*
                    Read from the InputStream from Arduino until termination character is reached.
                    Then send the whole String message to GUI Handler.
                     */
                    buffer[bytes] = (byte) mmInStream.read();
                    String readMessage;
                    if (buffer[bytes] == 'r'){
                        readMessage = new String(buffer,0,bytes-1);
                        Log.e("Msg : ",readMessage);
                        handler.obtainMessage(R_VALUE,readMessage).sendToTarget();
                        bytes = 0;
                    }
                    if (buffer[bytes] == 'i'){
                        readMessage = new String(buffer,1,bytes-1);
                        Log.e("Msg : ",readMessage);
                        handler.obtainMessage(IR_VALUE,readMessage).sendToTarget();
                        bytes = 0;
                    }
                    if (buffer[bytes] == 'x'){
                        readMessage = new String(buffer,1,bytes-1);
                        Log.e("Msg : ",readMessage);
                        handler.obtainMessage(X_VALUE,readMessage).sendToTarget();
                        bytes = 0;
                    }
                    if (buffer[bytes] == 'y'){
                        readMessage = new String(buffer,1,bytes-1);
                        Log.e("Msg : ",readMessage);
                        handler.obtainMessage(Y_VALUE,readMessage).sendToTarget();
                        bytes = 0;
                    }
                    if (buffer[bytes] == 'z'){
                        readMessage = new String(buffer,1,bytes-1);
                        Log.e("Msg : ",readMessage);
                        handler.obtainMessage(Z_VALUE,readMessage).sendToTarget();
                        bytes = 0;
                    }
                    if (buffer[bytes] == '\n'){
                        readMessage = new String(buffer,0,bytes);
                        Log.e("Msg : ",readMessage);
                        handler.obtainMessage(NEW_SEQUENCE,readMessage).sendToTarget();
                        bytes = 0;
                        i = i+1;
                    }
                    else {
                        bytes++;
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                    break;
                }
            }
        }

        /* Call this from the main activity to send data to the remote device */
        public void write(String input) {
            byte[] bytes = input.getBytes(); //converts entered String into bytes
            try {
                mmOutStream.write(bytes);
            } catch (IOException e) {
                Log.e("Send Error","Unable to send data",e);
            }
        }

        public int read_two_bytes(){
            byte[] b = new byte[1024];
            int intByte = 0;
            try {
                int bytesRead = mmInStream.read(b, 0, 2);
                intByte = b[1]*100+b[0];
            } catch (IOException e) {
                Log.e("Send Error","Unable to receive data", e);
            }
            return intByte;
        }

        /* Call this from the main activity to shutdown the connection */
        public void cancel() {
            try {
                mmSocket.close();
            } catch (IOException e) { }
        }
    }

    /* ============================ Terminate Connection at BackPress ====================== */
    @Override
    public void onBackPressed() {
        // Terminate Bluetooth Connection and close app
        if (createConnectThread != null){
            createConnectThread.cancel();
        }
        Intent a = new Intent(Intent.ACTION_MAIN);
        a.addCategory(Intent.CATEGORY_HOME);
        a.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(a);
    }

    /* ============================ Keep Receiving Bluetooth Data ============================= */
    public class ReceivingThread extends Thread{
        TextView value = findViewById(R.id.rValue);

        public void run(){
            while(true) {
                if(run){
                    runOnUiThread(new Thread(new Runnable() {
                        @Override
                        public void run() {
                            intBytes = connectedThread.read_two_bytes();
                            value.setText("Value : " + intBytes);
                        }
                    }));
                    try {
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                else{
                    runOnUiThread(new Thread(new Runnable() {
                        @Override
                        public void run() {
                            value.setText("Click Receive button to start");
                        }
                    }));
                    try {
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

    }

    public void HARPrediction(){
        if(az.size() > 0 && ppg1.size()%N_Samples == 0 && ppg2.size()%N_Samples == 0 && ax.size()%N_Samples == 0 && ay.size()%N_Samples == 0 && az.size()%N_Samples == 0){
            //Segment and Reshape Data into fixed window sizes

            float[] temp_ax = new float[100];
            float[] temp_ay = new float[100];
            float[] temp_az = new float[100];

            for(int pj=0; pj<temp_ax.length; pj++){
                temp_ax[pj] = toFloatArray(ax)[k*N_Samples+pj];
                temp_ay[pj] = toFloatArray(ay)[k*N_Samples+pj];
                temp_az[pj] = toFloatArray(az)[k*N_Samples+pj];
            }

            float[][][] input_3d = new float[1][N_Samples][3];
            for (int n = 0; n < N_Samples; n++){
                input_3d[0][n][0] = temp_ax[n];
                input_3d[0][n][1] = temp_ay[n];
                input_3d[0][n][2] = temp_az[n];
            }
            //Make predictions on input data window in HAR Classifier
            results = classifier.predictions(input_3d);

            //Output predictions to app UI
            restingTextView.setText("R : \t" + round((results[0][4] + results[0][1] + results[0][0]), 3));
            norestingTextView.setText("N : \t" + round((results[0][2] + results[0][3] + results[0][5]), 3));

            k++;
        }
    }

    public void HRProcessing(){
        if (ppg1.size()>0 && ppg1.size()==ax.size() && (ppg1.size()%(window*fs) == 0) && (ppg2.size()%(window*fs) == 0) && (ax.size()%(window*fs) == 0) && (ay.size()%(window*fs) == 0) && (az.size()%(window*fs) == 0)){
            ind = j*DK;
            fin = ind+K;

            Integer[] arr_PPG1 = ppg1.toArray(new Integer[0]);
            Integer[] arr_PPG2 = ppg2.toArray(new Integer[0]);
            Float[] arr_ax = ax.toArray(new Float[0]);
            Float[] arr_ay = ay.toArray(new Float[0]);
            Float[] arr_az = az.toArray(new Float[0]);

            Python py = Python.getInstance();
            final PyObject pyobj = py.getModule("HREstimation_v5");
            PyObject obj1 = pyobj.callAttr("main", fin, ind, j, arr_PPG1, arr_PPG2, heartbeat, arr_ax, arr_ay, arr_az);

            nilaiBPM = obj1.toString();

            detakJantung.post(new Runnable() {
                @Override
                public void run() {
                    detakJantung.setText("BPM : \t"+obj1.toString());
                }
            });

            heartbeat = Integer.valueOf(nilaiBPM);

            if(round((results[0][4] + results[0][1] + results[0][0]), 3) >= 0.500){
                counter++;
                movingAvrg = ((movingAvrg*(counter-1))+heartbeat)/counter;
            }

            if(counter >= 100){
                if(movingAvrg > 100 || movingAvrg < 60){
                    abnormalitas.setText("Normal : \t"+ "N");
                }
                else{
                    abnormalitas.setText("Normal : \t"+ "Y");
                }
            }
            else{
                abnormalitas.setText("Normal : ");
            }

            j++;

        }
    }

    private float[] toFloatArray(List<Float> list){
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list){
            array[i++] = (f != null ? f :Float.NaN);
        }
        return array;
    }

    //Rounds the output predictions to two decimal places
    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }

}