import SwiftUI

struct Prediction: Decodable {
    let x: Double; let y: Double; let confidence: Double; let uncertainty_px: Double; let warning: String
}

struct ContentView: View {
    @State private var image: UIImage?
    @State private var result: Prediction?
    @State private var showingPicker = false
    @State private var error = ""
    let apiURL = URL(string: "http://127.0.0.1:8000/predict")!

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if let image { Image(uiImage: image).resizable().scaledToFit().frame(maxHeight: 360) }
                Button("Choose image") { showingPicker = true }
                Button("Analyze locally") { Task { await predict() } }.disabled(image == nil)
                if let result { Text(String(format: "x %.1f, y %.1f\nuncertainty %.1f px\nconfidence %.2f", result.x, result.y, result.uncertainty_px, result.confidence)).multilineTextAlignment(.center) }
                if !error.isEmpty { Text(error).foregroundStyle(.red) }
                Text("Estimate only. No automatic BOTB submission.").font(.footnote)
            }.padding().navigationTitle("Ball Finder")
        }.sheet(isPresented: $showingPicker) { ImagePicker(image: $image) }
    }
    func predict() async {
        guard let image, let data = image.jpegData(compressionQuality: 0.95) else { return }
        var request = URLRequest(url: apiURL); request.httpMethod = "POST"; let boundary = UUID().uuidString; request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".utf8) + data + Data("\r\n--\(boundary)--\r\n".utf8)
        do { let (data, _) = try await URLSession.shared.data(for: request); result = try JSONDecoder().decode(Prediction.self, from: data) } catch { error = error.localizedDescription }
    }
}
