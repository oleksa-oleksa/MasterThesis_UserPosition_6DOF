using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System;
using System.Text;
using System.IO;


public class Camera2CSV : MonoBehaviour
{
    // Camera data
    public static Vector3 headposition;
    public static Quaternion orientation;
    public static Vector3 orientationEulerAngles;
    public static Vector3 velocity;
    public static double speed;

    // for CSV Logger
    private string filePath;
    private StreamWriter outStream;


    // Following method is used to retrive the relative path as device platform
    private string getPath()
    {
#if UNITY_EDITOR
        Directory.CreateDirectory(Application.dataPath + "/CSV");
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

    }


    void Update()
    {

        headposition = Camera.main.transform.position;
        orientation = Camera.main.transform.rotation;
        
        // additional camera data
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
        float velocity_x = velocity.x;
        float velocity_y = velocity.y;
        float velocity_z = velocity.z;

        // save camera data from current frame in CSV file
        WriteFrameToCSV(timestamp, x, y, z, qx, qy, qz, qw, velocity_x, velocity_y, velocity_z, (float)speed);
   
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

        outStream.WriteLine("timestamp,x,y,z,qx,qy,qz,qw,velocity_x,velocity_y,velocity_z,speed");
        outStream.Flush();

    }

    void WriteFrameToCSV(params float[] cameraData)
    {

        string delimiter = ",";
        string data = string.Join(delimiter, cameraData);

        outStream.WriteLine(data);
        outStream.Flush();
    }



    public void OnDestroy()
    {
        // OnDestroy occurs when a Scene or game ends.
        // If a Scene is closed and a new Scene is loaded the OnDestroy call will be made.
        // When built as a standalone application OnDestroy calls are made when Scenes end.

        // Closes the current StreamWriter object and the underlying stream.
        outStream.Close();

    }

}
