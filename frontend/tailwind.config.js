/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ember: {
          500: "#f97316",
          600: "#ea580c",
          700: "#c2410c"
        }
      }
    }
  },
  plugins: []
};
