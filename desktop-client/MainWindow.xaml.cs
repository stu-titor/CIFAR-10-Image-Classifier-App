using System.Text;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Net.Http;
using System.Text.Json;
using Microsoft.Win32;
using System.Runtime.CompilerServices;

namespace CifarInterface
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>

    public partial class MainWindow : Window
    {
        private static readonly HttpClient client = new HttpClient();
        private string imageInput;
        private bool? isImageFile;

        public MainWindow()
        {
            imageInput = "";
            isImageFile = null;
            InitializeComponent();
        }

        private void TextUrlFocused(object sender, RoutedEventArgs e)
        {
            if(TxtUrl.Text.Equals("Or paste an image URL here"))
            {
                TxtUrl.Text = "";
                TxtUrl.Foreground = System.Windows.Media.Brushes.Black;
            }
        }

        private void TextUrlUnfocused(object sender, RoutedEventArgs e)
        {
            if(TxtUrl.Text.Equals(""))
            {
                TxtUrl.Text = "Or paste an image URL here";
                TxtUrl.Foreground = System.Windows.Media.Brushes.Gray;
            }

        }

        private void TxtUrlType(object sender, TextChangedEventArgs e)
        {
            if(!TxtUrl.Text.Equals("Or paste an image URL here") && !TxtUrl.Text.Equals("")){
                imageInput = TxtUrl.Text;
                isImageFile = false;
            }
        }

        private void BtnPickFile_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog fileSelector = new OpenFileDialog();
            fileSelector.Filter = "Image Files (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp";
            fileSelector.Title = "Please select an image";

            bool? isSuccess = fileSelector.ShowDialog();
            if(isSuccess == true) {
                imageInput = fileSelector.FileName; //full path including file name
                isImageFile = true;
            }
        }

        private async void BtnClassify_Click(object sender, RoutedEventArgs e)
        {
            HttpResponseMessage response;

            if (isImageFile == null) {
                TxtResult.Text = "Please select an image first";
                return;
            } else if (isImageFile == true) {
                byte[] imageBytes = File.ReadAllBytes(imageInput);

                var content = new MultipartFormDataContent();
                var imageContent = new ByteArrayContent(imageBytes);
                content.Add(imageContent, "image", "upload.jpg");

                response = await client.PostAsync("http://3.228.219.225:5000/classify/file", content);
            } else {
                var payload = new { url = imageInput };
                string jsonVer = JsonSerializer.Serialize(payload);

                var content = new StringContent(jsonVer, Encoding.UTF8, "application/json");

                response = await client.PostAsync("http://3.228.219.225:5000/classify/url", content);
            }

            string result = await response.Content.ReadAsStringAsync();
            var json = JsonDocument.Parse(result);
            string prediction = json.RootElement.GetProperty("prediction").GetString() ?? "unknown";
            TxtResult.Text = $"This image contains a {prediction}";
        }
    }
}