/*
** EPITECH PROJECT, 2025
** AREA
** File description:
** vite.config
*/

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
    plugins: [react()],
});
