// require('dotenv').config({path:'./env'})

import dotenv from "dotenv";
import connectDB from "./db/index.js";
import { app } from "./app.js";
import os from "os";
import path from "path";
import express from "express";

dotenv.config({ path: "./env" });

const _dirname = path.dirname("");
const frontendBP = path.join(_dirname, "../Frontend/dist");
app.use(express.static(frontendBP));

connectDB()
  .then(() => {
    app.on("error", (error) => {
      console.log("ERROR: ", error);
      throw error;
    });
    app.listen(process.env.PORT || 8000, () => {
      console.log(`Server is running at port : ${process.env.PORT}`);
    });
  })
  .catch((err) => {
    console.log("MONGODB connection failed !!!", err);
  });
