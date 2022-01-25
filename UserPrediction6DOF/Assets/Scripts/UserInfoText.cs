using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System;
using System.Text;
using System.IO;


public class UserInfoText : MonoBehaviour
{
    public GameObject textDisplay;
    public TextMeshPro textmeshPro;

    // Camera 
    public Vector3 headposition;
    public Quaternion orientation;
    public Vector3 orientationEulerAngles;
    public Vector3 velocity;
    public double speed;

    // time variables for delay
    private float waitTime = 1.0f;
    private float timer = 0.0f;

    // for CSV Logger
    private string filePath;
    private StreamWriter outStream;


    // Following method is used to retrive the relative path as device platform
    private string getPath()
    {
#if UNITY_EDITOR
        return Application.dataPath +"/CSV/"+"Saved_data.csv";
#elif UNITY_ANDROID
        return Application.persistentDataPath+"Saved_data.csv";
#elif UNITY_IPHONE
        return Application.persistentDataPath+"/"+"Saved_data.csv";
#elif WINDOWS_UWP
        return Application.persistentDataPath  + "/CSV/";
#else
        return Application.dataPath + "/CSV/";
#endif
    }


    void Start()
    {
      
        OpenCSVFile();
        WriteHeaderToCSV();



        // original 
        //Save();
    }

  
    void Update()
    {
        
        timer += Time.deltaTime;
        
        headposition = Camera.main.transform.position;
        orientation = Camera.main.transform.rotation;
        orientationEulerAngles = Camera.main.transform.eulerAngles;
        velocity = Camera.main.velocity;
        speed = Math.Sqrt(Math.Pow(velocity[0], 2) + Math.Pow(velocity[1], 2) + Math.Pow(velocity[2], 2));

        // timestamp,x,y,z,qx,qy,qz,qw - structure for CSV
        // creates values for CSV
        float timestamp = Time.time;
        float x = headposition.x;
        float y = headposition.y;
        float z = headposition.z;
        float qx = orientation.x;
        float qy = orientation.y;
        float qz = orientation.z;
        float qw = orientation.w;


        // Check if we have reached the time limit
        if (timer > waitTime)
        {

            // save camera data from current frame in CSV file
            WriteFrameToCSV(timestamp, x, y, z, qx, qy, qz, qw);

            // update text block in hologram 
            textmeshPro = GetComponent<TextMeshPro>();
            textmeshPro.SetText("Position x: {0:3}, y: {1:3}, z: {2:3}; \r\n " +
                "Velocity x: {3:3}, y: {4:3}, z: {5:3}; \r\n" +
                "Speed: {6:3}",
               x, y, z, velocity[0], velocity[1], velocity[2], (float)speed);

            // reset timer
            timer = 0.0f;

        }
    }


    void OpenCSVFile() 
    {
        //  retriving the relative path as device platform
       filePath = getPath() + DateTime.Now.ToString("yyyyMMdd_HHmm") + "_saved.csv";

        //  Creates or opens a file for writing UTF-8 encoded text.
        //  If the file already exists, its contents are overwritten.
        //  By using a timestamp in file name it is supposed to creare a new csv file
        //  for every start of HoloLens Application
        outStream = System.IO.File.CreateText(filePath);
    }

    void WriteHeaderToCSV()
    {

        outStream.WriteLine("timestamp,x,y,z,qx,qy,qz,qw");
        outStream.Flush();

    }

   void WriteFrameToCSV(params float[] cameraData)
    {

        string delimiter = ", "; // comma with space after 
        string data = string.Join(delimiter, cameraData);

        outStream.WriteLine(data);
        outStream.Flush();
    }


    void WriteHeaderToCSV_OLD()
    {

        //  retriving the relative path as device platform
        filePath = getPath() + DateTime.Now.ToString("yyyyMMdd_HHmm") + "_saved.csv";

        //  Creates or opens a file for writing UTF-8 encoded text.
        //  If the file already exists, its contents are overwritten.
        //  By using a timestamp in file name it is supposed to creare a new csv file
        //  for every start of HoloLens Application
        outStream = System.IO.File.CreateText(filePath);

        string header = "timestamp,x,y,z,qx,qy,qz,qw";
        outStream.WriteLine(header);

        outStream.Flush();

        string tmp = "newnewnew";
        outStream.WriteLine(tmp);

        outStream.Close();

        /*
        outStream = System.IO.File.CreateText(filePath);
        string tmp = "4535435454354";
        outStream.WriteLine(tmp);
        */
    }

    public void OnDestroy()
    {
        // OnDestroy occurs when a Scene or game ends.
        // If a Scene is closed and a new Scene is loaded the OnDestroy call will be made.
        // When built as a standalone application OnDestroy calls are made when Scenes end.

        // Closes the current StreamWriter object and the underlying stream.
        outStream.Close();

    }

    /*
    void Save()
    {

        // Creating First row of titles manually..
        string[] rowDataTemp = new string[3];
        rowDataTemp[0] = "Name";
        rowDataTemp[1] = "ID";
        rowDataTemp[2] = "Income";
        rowData.Add(rowDataTemp);

        // You can add up the values in as many cells as you want.
        for (int i = 0; i < 10; i++)
        {
            rowDataTemp = new string[3];
            rowDataTemp[0] = "Alexandra" + i; // name
            rowDataTemp[1] = "" + i; // ID
            rowDataTemp[2] = "$" + UnityEngine.Random.Range(5000, 10000); // Income
            rowData.Add(rowDataTemp);
        }

        string[][] output = new string[rowData.Count][];

        for (int i = 0; i < output.Length; i++)
        {
            output[i] = rowData[i];
        }

        int length = output.GetLength(0);
        string delimiter = ",";

        StringBuilder sb = new StringBuilder();

        for (int index = 0; index < length; index++)
            sb.AppendLine(string.Join(delimiter, output[index]));


        string filePath = getPath();

        StreamWriter outStream = System.IO.File.CreateText(filePath);
        outStream.WriteLine(sb);
        outStream.Close();
    }

    */
}
