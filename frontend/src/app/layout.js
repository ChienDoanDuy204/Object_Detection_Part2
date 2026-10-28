import "./globals.css";

export const metadata = {
  title: "Faster R-CNN Studio | Interactive Object Detection with FPS Control",
  description: "End-to-End Faster R-CNN with ResNet-18 Backbone and RPN. Direct manipulation interface for Photos, Videos, and Live Camera Streams with customizable processing FPS.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
