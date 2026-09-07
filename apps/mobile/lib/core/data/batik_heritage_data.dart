import 'package:flutter/material.dart';
import 'motif_asset_registry.dart';

/// BatikHeritageItem holds factual cultural, historical, and regional metadata for a batik motif.
class BatikHeritageItem {
  final String id;
  final List<String> rawLabels;
  final String name;
  final String region;
  final String province;
  final String island;
  final String category; // Keraton, Pesisir, Pedalaman, Kontemporer
  final String shortDescription;
  final String philosophy;
  final String history;
  final String visualCharacteristics;
  final String usageOccasion;
  final Color primaryColor;
  final Color secondaryColor;
  final String tag;

  /// Curated photographic batik image asset path (or resolved via MotifAssetRegistry).
  String? get imagePath => MotifAssetRegistry.getAssetPath(id);

  /// Whether this motif has a photographic asset available.
  bool get hasImageAsset => imagePath != null;

  const BatikHeritageItem({
    required this.id,
    required this.rawLabels,
    required this.name,
    required this.region,
    required this.province,
    required this.island,
    required this.category,
    required this.shortDescription,
    required this.philosophy,
    required this.history,
    required this.visualCharacteristics,
    required this.usageOccasion,
    required this.primaryColor,
    required this.secondaryColor,
    required this.tag,
  });

  bool matches(String query) {
    final q = query.toLowerCase();
    return name.toLowerCase().contains(q) ||
        region.toLowerCase().contains(q) ||
        province.toLowerCase().contains(q) ||
        island.toLowerCase().contains(q) ||
        category.toLowerCase().contains(q) ||
        tag.toLowerCase().contains(q);
  }
}

/// Static educational database of Indonesian batik motifs.
class BatikHeritageData {
  static const List<BatikHeritageItem> allMotifs = [
    BatikHeritageItem(
      id: 'parang_rusak',
      rawLabels: ['batik-parang', 'batik-parang-rusak', 'parang', 'Parang_Barong'],
      name: 'Parang Rusak',
      region: 'Yogyakarta & Surakarta',
      province: 'D.I. Yogyakarta & Jawa Tengah',
      island: 'Jawa',
      category: 'Keraton (Larangan)',
      shortDescription: 'Simbol ketabahan manusia dalam mengendalikan hawa nafsu dan perjuangan tanpa henti.',
      philosophy:
          'Bentuk geometris miring menyerupai ombak karang laut selatan melambangkan semangat pantang menyerah, keteguhan hati, kesinambungan budi pekerti luhur, dan kewibawaan kepemimpinan.',
      history:
          'Diciptakan oleh Sultan Agung Hanyokrokusumo saat bertapa di pesisir Pantai Selatan (Parangtritis). Dahulu merupakan motif larangan (Batik Larangan) yang hanya boleh dikenakan oleh raja dan bangsawan keraton.',
      visualCharacteristics:
          'Pola garis diagonal miring berulang (lereng) yang membentuk jajaran huruf "S" sambung-menyambung dengan aksen mlinjon di sela garis.',
      usageOccasion: 'Upacara adat keraton, prosesi sakral, dan pakaian kebesaran raja/bangsawan.',
      primaryColor: Color(0xFF5A2A18),
      secondaryColor: Color(0xFFC8963E),
      tag: 'Batik Larangan',
    ),
    BatikHeritageItem(
      id: 'megamendung',
      rawLabels: ['batik-megamendung', 'megamendung'],
      name: 'Megamendung',
      region: 'Cirebon',
      province: 'Jawa Barat',
      island: 'Jawa',
      category: 'Pesisir (Akulturasi)',
      shortDescription: 'Visualisasi awan pembawa hujan penyejuk kehidupan berpadu dengan akulturasi Tionghoa.',
      philosophy:
          'Awan mendung yang menaungi bumi bermakna kesabaran, kepala dingin, dan ketenangan jiwa dalam menghadapi segala permasalahan hidup agar membawa kesuburan dan keberkahan bagi sekitar.',
      history:
          'Lahir dari akulturasi budaya keraton Cirebon dengan tradisi Tionghoa ketika Sunan Gunung Jati menikahi Putri Ong Tien dari Tiongkok pada abad ke-16.',
      visualCharacteristics:
          'Garis lengkung awan horizontal dengan gradasi warna 7 tingkatan (gradasi lapis saphire/biru atau merah) yang dinamis meruncing.',
      usageOccasion: 'Busana resmi, upacara adat pesisir, perayaan seni, dan busana kontemporer.',
      primaryColor: Color(0xFF1D3557),
      secondaryColor: Color(0xFF457B9D),
      tag: 'Warisan Cirebon',
    ),
    BatikHeritageItem(
      id: 'kawung',
      rawLabels: ['batik-kawung', 'kawung'],
      name: 'Kawung',
      region: 'Yogyakarta & Surakarta',
      province: 'D.I. Yogyakarta & Jawa Tengah',
      island: 'Jawa',
      category: 'Keraton (Geometris)',
      shortDescription: 'Simbol kesucian hati, keadilan, serta asal-usul kehidupan manusia yang seimbang.',
      philosophy:
          'Bentuk lonjong empat penjuru mata angin bermakna konsep sedulur papat lima pancer, melambangkan manusia yang mampu mengendalikan empat nafsu untuk mencapai kesucian batin dan keadilan moral.',
      history:
          'Merupakan salah satu motif tertua di Nusantara yang telah terdokumentasi sejak abad ke-13 pada relief candi Jawa Kuno (Candi Prambanan dan Candi Panataran).',
      visualCharacteristics:
          'Bentuk elips menyerupai buah kolang-kaling atau daun teratai yang tersusun simetris bersilangan membentuk pola lingkaran harmonis.',
      usageOccasion: 'Upacara kenegaraan, prosesi pernikahan adat Jawa, dan peribadatan formal.',
      primaryColor: Color(0xFF6B3E26),
      secondaryColor: Color(0xFFD4A373),
      tag: 'Klasik Mataram',
    ),
    BatikHeritageItem(
      id: 'bali_barong',
      rawLabels: ['Bali_Barong', 'batik-bali', 'bali'],
      name: 'Bali Barong',
      region: 'Denpasar & Gianyar',
      province: 'Bali',
      island: 'Bali',
      category: 'Pesisir & Sakral',
      shortDescription: 'Epitom pelindung spiritual dan simbol kemenangan kebajikan (Dharma) melawan kejahatan.',
      philosophy:
          'Figur Barong sebagai simbol Dharma merepresentasikan kekuatan magis pelindung manusia dan penyeimbang keharmonisan alam semesta (Tri Hita Karana).',
      history:
          'Berkembang di pusat seni Gianyar dan Denpasar, memadukan teknik canting Jawa dengan kekayaan ornamen ukiran pura dan mitologi Hindu Bali.',
      visualCharacteristics:
          'Figur kepala Barong megah dengan taring dan ornamen pepatran (sulur daun kamboja, teratai) dengan warna cerah dan kontras ekspresif.',
      usageOccasion: 'Upacara keagamaan Hindu Bali, festival seni budaya, dan busana pesta adat.',
      primaryColor: Color(0xFF8B1E1E),
      secondaryColor: Color(0xFFDDA15E),
      tag: 'Khas Pulau Dewata',
    ),
    BatikHeritageItem(
      id: 'sekar_jagad',
      rawLabels: ['batik-sekar', 'batik-sekar-jagad', 'sekar_jagad'],
      name: 'Sekar Jagad',
      region: 'Yogyakarta & Solo',
      province: 'D.I. Yogyakarta & Jawa Tengah',
      island: 'Jawa',
      category: 'Keraton (Botanikal)',
      shortDescription: 'Bunga keindahan dunia yang merangkum keanekaragaman flora nusantara yang memikat.',
      philosophy:
          'Nama "Sekar Jagad" berasal dari kata "kar" (peta/keindahan) dan "jagad" (dunia). Melambangkan keindahan keberagaman suku bangsa dan budaya dalam satu kesatuan yang harmonis.',
      history:
          'Muncul sejak masa Kesultanan Mataram Islam abad ke-18 sebagai manifestasi rasa syukur terhadap kemakmuran bumi Nusantara.',
      visualCharacteristics:
          'Pola pulau-pulau kecil asimetris (peta) yang di dalamnya memuat beragam isen-isen bunga, dedaunan, dan ornamen batik mini.',
      usageOccasion: 'Busana pernikahan (pakaian orang tua pengantin), wisuda, dan acara kenegaraan.',
      primaryColor: Color(0xFF783D19),
      secondaryColor: Color(0xFFE9C46A),
      tag: 'Simbol Keindahan',
    ),
    BatikHeritageItem(
      id: 'pintu_aceh',
      rawLabels: ['Aceh_Pintu_Aceh', 'pintu_aceh', 'batik-aceh'],
      name: 'Pintu Aceh',
      region: 'Banda Aceh',
      province: 'Aceh',
      island: 'Sumatera',
      category: 'Pesisir (Arsitektural)',
      shortDescription: 'Cerminan kepribadian masyarakat Aceh yang santun, ramah, namun teguh memegang prinsip.',
      philosophy:
          'Mengambil inspirasi dari pintu rumah adat Aceh yang rendah mengharuskan tamu menunduk, bermakna kerendahan hati, adab sopan santun, dan ketakwaan religius.',
      history:
          'Awalnya merupakan motif perhiasan logam emas peninggalan Sultan Iskandar Muda yang kemudian diadaptasikan ke seni wastra batik modern Aceh pada dekade 1970-an.',
      visualCharacteristics:
          'Siluet gerbang pintu ramping dengan ornamen sulur geometris teratur yang rapat dan bernuansa islami tanpa figur makhluk bernyawa.',
      usageOccasion: 'Pesta adat Meukuta, resepsi pernikahan formal, dan busana kerja kelembagaan.',
      primaryColor: Color(0xFF2A5235),
      secondaryColor: Color(0xFFE0A96D),
      tag: 'Serambi Mekkah',
    ),
    BatikHeritageItem(
      id: 'kalimantan_dayak',
      rawLabels: ['Kalimantan_Dayak', 'dayak', 'batik-dayak'],
      name: 'Dayak Batang Garing',
      region: 'Palangka Raya & Samarinda',
      province: 'Kalimantan Tengah & Timur',
      island: 'Kalimantan',
      category: 'Pedalaman & Kosmologis',
      shortDescription: 'Pohon kehidupan penopang jagat raya dan hubungan sakral manusia dengan alam leluhur.',
      philosophy:
          'Pohon Batang Garing melambangkan tiga alam (dunia atas, dunia tengah tempat manusia berakar, dan dunia bawah), mengajarkan kelestarian ekosistem hutan dan penghormatan leluhur.',
      history:
          'Diwariskan secara turun-temurun oleh suku Dayak Ngaju dan Kenyah, diabadikan dalam kain tenun dan seni lukis tameng perang sebelum diadaptasikan menjadi seni batik.',
      visualCharacteristics:
          'Lengkungan sulur meliuk dinamis (kaligrafi Dayak), motif tameng perisai (Talawang), dan burung Enggang gading dengan warna merah, kuning, dan hitam pekat.',
      usageOccasion: 'Ritual adat Tiwah, festival Erau, upacara perkawinan adat, dan perayaan panen.',
      primaryColor: Color(0xFF9E2A2B),
      secondaryColor: Color(0xFFF4A261),
      tag: 'Kearifan Borneo',
    ),
    BatikHeritageItem(
      id: 'papua_cendrawasih',
      rawLabels: ['Papua_Cendrawasih', 'Papua_Tifa', 'cendrawasih', 'batik-papua'],
      name: 'Papua Cendrawasih & Tifa',
      region: 'Jayapura & Asmat',
      province: 'Papua',
      island: 'Papua',
      category: 'Kontemporer Etnik',
      shortDescription: 'Pesona burung surga dan instrumen magis pengobar semangat persaudaraan bumi cenderawasih.',
      philosophy:
          'Burung Cendrawasih melambangkan keanggunan, keelokan alami, dan kedamaian, sedangkan Tifa melambangkan ketegasan pemimpin dalam memanggil persatuan warga.',
      history:
          'Diperkenalkan dan dikembangkan sejak tahun 1985 melalui program pertukaran perajin batik Solo ke Papua, melahirkan sintesis seni canting dengan ornamen ukir Asmat dan Sentani.',
      visualCharacteristics:
          'Figur burung berekor panjang melayang anggun dipadu ornamen tifa berkulit biawak, ukiran geometris spiral Asmat, dan warna alam tanah terakota-emas cerah.',
      usageOccasion: 'Festival Lembah Baliem, upacara kedatangan tamu kehormatan, dan busana formal Nusantara.',
      primaryColor: Color(0xFF5E2B6D),
      secondaryColor: Color(0xFFE76F51),
      tag: 'Permata Timur',
    ),
    BatikHeritageItem(
      id: 'pekalongan_tujuh_rupa',
      rawLabels: ['batik-pekalongan', 'pekalongan'],
      name: 'Pekalongan Tujuh Rupa',
      region: 'Pekalongan',
      province: 'Jawa Tengah',
      island: 'Jawa',
      category: 'Pesisir (Multikultural)',
      shortDescription: 'Ledakan warna warni pesisir yang dinamis mencerminkan keterbukaan dan kebebasan berekspresi.',
      philosophy:
          'Keanekaragaman 7 ragam flora dan fauna dalam satu helai kain melambangkan keluwesan beradaptasi, keberanian berinovasi, dan harmoni pergaulan antarbangsa.',
      history:
          'Pekalongan sebagai pelabuhan niaga strategis menyerap budaya Arab, Cina, India, dan Belanda (Batik Encim & Batik Belanda) yang melahirkan ragam hias paling ekspresif di Nusantara.',
      visualCharacteristics:
          'Warna-warna cerah ceria (merah muda, hijau pupus, toska), gambar buketan bunga Eropa, burung merak, dan kupu-kupu dengan isen cecek yang sangat halus.',
      usageOccasion: 'Busana pesta santai, resepsi pernikahan modern, dan kain sarung harian.',
      primaryColor: Color(0xFF264653),
      secondaryColor: Color(0xFF2A9D8F),
      tag: 'Kota Batik Dunia',
    ),
    BatikHeritageItem(
      id: 'lasem_tiga_negeri',
      rawLabels: ['batik-lasem', 'lasem'],
      name: 'Lasem Tiga Negeri',
      region: 'Rembang',
      province: 'Jawa Tengah',
      island: 'Jawa',
      category: 'Pesisir (Akulturasi Sejati)',
      shortDescription: 'Mahakarya perpaduan warna merah darah ayam Tionghoa, biru pekalongan, dan cokelat soga keraton.',
      philosophy:
          'Tiga warna pokok melambangkan toleransi, persaudaraan lintas etnis Tionghoa dan Jawa, serta keselarasan hidup dalam keragaman budaya.',
      history:
          'Proses pewarnaan legendaris yang dahulu dikerjakan di tiga kota berbeda: Merah di Lasem, Biru di Pekalongan/Kudus, dan Cokelat Soga di Solo/Yogyakarta.',
      visualCharacteristics:
          'Motif burung Hong (Phoenix), naga (Liong), dan bunga teratai berpadu dengan motif parang dan sekar jagad Jawa yang klasik.',
      usageOccasion: 'Pernikahan peranakan Tionghoa-Jawa, perayaan Imlek, dan koleksi adibusana bernilai tinggi.',
      primaryColor: Color(0xFF9E0031),
      secondaryColor: Color(0xFFC8963E),
      tag: 'Pusaka Tiga Negeri',
    ),
    BatikHeritageItem(
      id: 'minang_rangkiang',
      rawLabels: ['Sumatera_Barat_Rumah_Minang', 'rumah_minang', 'minang'],
      name: 'Minangkabau Rangkiang',
      region: 'Padang & Bukittinggi',
      province: 'Sumatera Barat',
      island: 'Sumatera',
      category: 'Pesisir (Filosofi Adat)',
      shortDescription: 'Simbol lumbung pangan keluarga, kesejahteraan masyarakat, dan ketahanan kaum.',
      philosophy:
          'Rangkiang sebagai tempat penyimpanan padi merepresentasikan kearifan hidup berhemat, kesiapan menghadapi masa sulit, dan kemakmuran bersama yang berkeadilan.',
      history:
          'Terinspirasi dari ukiran kayu dinding Rumah Gadang (Itiak Pulang Patang, Kaluak Paku) yang diintegrasikan ke tradisi batik tanah liek Sumatera Barat.',
      visualCharacteristics:
          'Siluet atap gonjong rumah gadang bertingkat dipadu ukiran kaluak paku melingkar simetris dengan pewarnaan alami tanah liat hangat.',
      usageOccasion: 'Upacara Batagak Gala, alek nagari, dan pakaian formal adat Minang.',
      primaryColor: Color(0xFF7A4419),
      secondaryColor: Color(0xFFE29578),
      tag: 'Kearifan Minang',
    ),
    BatikHeritageItem(
      id: 'pala_maluku',
      rawLabels: ['Maluku_Pala', 'pala', 'batik-maluku'],
      name: 'Maluku Buah Pala',
      region: 'Banda & Ambon',
      province: 'Maluku',
      island: 'Maluku',
      category: 'Kepulauan Rempah',
      shortDescription: 'Emas hitam aromatik yang mengubah sejarah perdagangan rempah jalur maritim dunia.',
      philosophy:
          'Buah pala melambangkan kekayaan alam yang bernilai tinggi, ketahanan jati diri rempah nusantara, dan keharuman budi pekerti warga Maluku.',
      history:
          'Kepulauan Banda adalah satu-satunya produsen pala asli dunia pada era kolonial. Motif ini merayakan warisan agung Kepulauan Rempah (Spice Islands).',
      visualCharacteristics:
          'Buah pala merekah memperlihatkan fuli merah menyala berpadu cengkih dan gelombang samudra biru kepulauan.',
      usageOccasion: 'Pesta adat Panas Pela, pesta rakyat Maluku, dan busana daerah formal.',
      primaryColor: Color(0xFF3D2645),
      secondaryColor: Color(0xFFE07A5F),
      tag: 'Warisan Rempah',
    ),
    BatikHeritageItem(
      id: 'batik_betawi',
      rawLabels: ['batik-betawi', 'betawi'],
      name: 'Batik Betawi Pucuk Rebung',
      region: 'Jakarta',
      province: 'DKI Jakarta',
      island: 'Jawa',
      category: 'Pesisir & Urban',
      shortDescription: 'Ragam hias cerah khas tanah Batavia dengan aksen segitiga pucuk rebung nan memesona.',
      philosophy: 'Pucuk rebung melambangkan cita-cita mulia yang terus bertumbuh tegak lurus menggapai kebajikan.',
      history: 'Berkembang di kawasan pesisir Batavia sejak abad ke-19, memadukan corak Sunda, Tionghoa, dan Arab.',
      visualCharacteristics: 'Warna-warna cerah seperti merah, kuning, dan hijau dengan pola geometris tumpal/rebung.',
      usageOccasion: 'Pesta pernikahan Betawi, festival Abang None, dan busana resmi daerah.',
      primaryColor: Color(0xFFB22222),
      secondaryColor: Color(0xFFF1C40F),
      tag: 'Budaya Betawi',
    ),
    BatikHeritageItem(
      id: 'batik_celup',
      rawLabels: ['batik-celup', 'celup', 'jumputan'],
      name: 'Batik Celup Jumputan',
      region: 'Yogyakarta & Surakarta',
      province: 'D.I. Yogyakarta & Jawa Tengah',
      island: 'Jawa',
      category: 'Ikat Celup Tradisional',
      shortDescription: 'Seni celup rintang tradisional dengan motif bintik-bintik melingkar yang dinamis dan anggun.',
      philosophy: 'Keteraturan ikatan yang menghasilkan pola indah melambangkan ketabahan menjalani dinamika kehidupan.',
      history: 'Metode tekstil kuno Nusantara yang dipengaruhi teknik tie-dye India dan Tiongkok sejak era Majapahit.',
      visualCharacteristics: 'Pola bintik berulang, gradasi celupan warna kontras cerah pada kain mori atau sutra.',
      usageOccasion: 'Selendang kebaya adat, busana santai tradisional, dan selendang tari.',
      primaryColor: Color(0xFF2E4057),
      secondaryColor: Color(0xFFF39C12),
      tag: 'Seni Jumputan',
    ),
    BatikHeritageItem(
      id: 'batik_ceplok',
      rawLabels: ['batik-ceplok', 'ceplok'],
      name: 'Batik Ceplok Kasunanan',
      region: 'Yogyakarta & Solo',
      province: 'Jawa Tengah & D.I. Yogyakarta',
      island: 'Jawa',
      category: 'Klasik Geometris',
      shortDescription: 'Keteraturan pola geometris roset kotak simetris penuntun hidup harmonis dan bersahaja.',
      philosophy: 'Melambangkan keteraturan kosmos, keseimbangan batin manusia, dan kesucian hati nurani.',
      history: 'Merupakan salah satu pola dasar tertua Mataram Islam abad ke-17 yang wajib dikuasai para pembatik keraton.',
      visualCharacteristics: 'Bentuk bintang, mawar, segi empat simetris dengan isen cecek soga dan nila indigo.',
      usageOccasion: 'Upacara midodareni, pakaian adat harian para sesepuh, dan pertemuan formal.',
      primaryColor: Color(0xFF3B2F2F),
      secondaryColor: Color(0xFFD4AF37),
      tag: 'Klasik Mataram',
    ),
    BatikHeritageItem(
      id: 'batik_ciamis',
      rawLabels: ['batik-ciamis', 'ciamis', 'ciamisan'],
      name: 'Batik Ciamisan Batu Ngampar',
      region: 'Ciamis',
      province: 'Jawa Barat',
      island: 'Jawa',
      category: 'Tatar Galuh',
      shortDescription: 'Kesederhanaan motif alam pesisir selatan Pasundan bernuansa tenang dan bersahaja.',
      philosophy: 'Menghargai alam ciptaan, keteguhan prinsip laksana batu sungai, dan ketenangan jiwa Sunda.',
      history: 'Lahir dari para pengungsi Perang Diponegoro abad ke-19 yang berbaur dengan kearifan lokal Tatar Galuh.',
      visualCharacteristics: 'Warna kalem didominasi hitam, putih, dan cokelat soga dengan ornamen flora sederhana.',
      usageOccasion: 'Busana formal Sunda, pertemuan resmi dinas, dan upacara adat Galuh.',
      primaryColor: Color(0xFF2C3539),
      secondaryColor: Color(0xFFC49A45),
      tag: 'Warisan Galuh',
    ),
    BatikHeritageItem(
      id: 'batik_garutan',
      rawLabels: ['batik-garutan', 'garutan', 'garut'],
      name: 'Batik Garutan Merak Ngibing',
      region: 'Garut',
      province: 'Jawa Barat',
      island: 'Jawa',
      category: 'Pasundan Priangan',
      shortDescription: 'Keanggunan tarian merak pesona tanah intan dengan warna gading gumading nan memikat.',
      philosophy: 'Merak menari melambangkan keelokan budi pekerti wanita Sunda, keceriaan hidup, dan keharmonisan.',
      history: 'Berkembang pesat sejak awal abad ke-20 berpusat di daerah Tarogong dan Kota Garut.',
      visualCharacteristics: 'Warna pastel khas (gumading/krem gading, merah bata, toska) dengan motif merak meliuk lincah.',
      usageOccasion: 'Kain panjang kebaya pesta, wisuda, dan acara seremonial kenegaraan.',
      primaryColor: Color(0xFF9E4770),
      secondaryColor: Color(0xFFF7B05B),
      tag: 'Pesona Garut',
    ),
    BatikHeritageItem(
      id: 'batik_gentongan',
      rawLabels: ['batik-gentongan', 'gentongan'],
      name: 'Batik Gentongan Madura',
      region: 'Bangkalan',
      province: 'Jawa Timur',
      island: 'Jawa',
      category: 'Pesisir Madura',
      shortDescription: 'Batik pusaka rendaman gentong tanah liat berbulan-bulan dengan warna magis tahan luntur.',
      philosophy: 'Ketabahan dan dedikasi luar biasa para perempuan Madura dalam melestarikan karya adiluhung.',
      history: 'Teknik rendaman di gentong gerabah selama 6 hingga 12 bulan yang diwariskan turun-temurun di Tanjungbumi.',
      visualCharacteristics: 'Warna merah cabe, kuning kunyit, dan ungu pekat yang meresap mendalam ke serat kain.',
      usageOccasion: 'Busana kehormatan adat Madura, koleksi kain antik bernilai tinggi, dan pusaka keluarga.',
      primaryColor: Color(0xFF800020),
      secondaryColor: Color(0xFFF4D03F),
      tag: 'Pusaka Madura',
    ),
    BatikHeritageItem(
      id: 'batik_keraton',
      rawLabels: ['batik-keraton', 'keraton'],
      name: 'Batik Keraton Lar Garuda',
      region: 'Surakarta & Yogyakarta',
      province: 'Jawa Tengah & D.I. Yogyakarta',
      island: 'Jawa',
      category: 'Keraton Larangan',
      shortDescription: 'Kewibawaan sayap burung garuda penjaga kedaulatan dan keagungan falsafah raja-raja Jawa.',
      philosophy: 'Sayap garuda (Mahameru) melambangkan kekuatan spiritual, kepemimpinan mulia, dan keadilan luhur.',
      history: 'Merupakan batik pola larangan (awisan dalem) yang dahulu eksklusif dikenakan para raja dan kerabat dekat.',
      visualCharacteristics: 'Sayap garuda tunggal atau sepasang, ornamen lidah api, dan latar soga cokelat hitam legam.',
      usageOccasion: 'Upacara penobatan raja, pernikahan agung keraton, dan resepsi kenegaraan resmi.',
      primaryColor: Color(0xFF4A2E18),
      secondaryColor: Color(0xFFD4AF37),
      tag: 'Pola Larangan',
    ),
    BatikHeritageItem(
      id: 'batik_priangan',
      rawLabels: ['batik-priangan', 'priangan'],
      name: 'Batik Priangan Tasikmalaya',
      region: 'Tasikmalaya',
      province: 'Jawa Barat',
      island: 'Jawa',
      category: 'Pasundan',
      shortDescription: 'Ragam flora krisan dan melati lereng Parahyangan yang lembut, asri, dan menyejukkan batin.',
      philosophy: 'Kesuburan tanah Priangan dan kelembutan tutur kata masyarakat Sunda yang mencintai kedamaian.',
      history: 'Dipengaruhi oleh corak Mataram yang diperhalus dengan estetika alam pegunungan Priangan Timur.',
      visualCharacteristics: 'Garis ornamen tipis luwes, cecek halus rapat, serta warna tanah berpadu ungu dan hijau pupus.',
      usageOccasion: 'Busana kebaya Sunda formal, resepsi adat, dan seragam resmi.',
      primaryColor: Color(0xFF335C67),
      secondaryColor: Color(0xFFE09F3E),
      tag: 'Nuansa Parahyangan',
    ),
    BatikHeritageItem(
      id: 'batik_sidoluhur',
      rawLabels: ['batik-sidoluhur', 'sidoluhur'],
      name: 'Batik Sido Luhur',
      region: 'Surakarta',
      province: 'Jawa Tengah',
      island: 'Jawa',
      category: 'Klasik Keraton',
      shortDescription: 'Doa suci agar pemakainya mencapai derajat keluhuran budi, martabat, dan kemuliaan hidup.',
      philosophy: 'Kata "Sido" berarti terwujud dan "Luhur" berarti berbudi mulia, doa bagi manusia agar berguna bagi semesta.',
      history: 'Diciptakan oleh Ki Ageng Henis pada abad ke-16 sebagai wasiat moral bagi keturunan bangsawan Jawa.',
      visualCharacteristics: 'Pola wajik beraturan memuat pohon hayat, sayap garuda, dan ornamen tahta bertatah soga cokelat.',
      usageOccasion: 'Prosesi ijab kabul pernikahan adat Jawa dan perayaan hari besar keluarga.',
      primaryColor: Color(0xFF532E1C),
      secondaryColor: Color(0xFFDAA520),
      tag: 'Doa Kemuliaan',
    ),
    BatikHeritageItem(
      id: 'batik_sidomukti',
      rawLabels: ['batik-sidomukti', 'sidomukti'],
      name: 'Batik Sido Mukti',
      region: 'Surakarta & Yogyakarta',
      province: 'Jawa Tengah',
      island: 'Jawa',
      category: 'Klasik Pernikahan',
      shortDescription: 'Pola agung pengantin Jawa pembawa berkah kebahagiaan sejati, rezeki melimpah, dan ketenteraman.',
      philosophy: 'Doa agar kedua mempelai memperoleh kemakmuran lahir dan batin serta senantiasa dilimpahi rezeki halal.',
      history: 'Batik pusaka yang wajib dikenakan sepasang pengantin pada upacara panggih pernikahan adat Jawa.',
      visualCharacteristics: 'Ornamen kupu-kupu, singgasana meru, dan ukiran bunga dalam bingkai belah ketupat soga berlatar ukel.',
      usageOccasion: 'Pakaian utama pengantin pria dan wanita saat upacara pernikahan adat.',
      primaryColor: Color(0xFF43281C),
      secondaryColor: Color(0xFFE6BA50),
      tag: 'Pusaka Pengantin',
    ),
    BatikHeritageItem(
      id: 'batik_sogan',
      rawLabels: ['batik-sogan', 'sogan'],
      name: 'Batik Sogan Klasik',
      region: 'Solo & Yogyakarta',
      province: 'Jawa Tengah & D.I. Yogyakarta',
      island: 'Jawa',
      category: 'Tradisi Pewarnaan Alam',
      shortDescription: 'Warna cokelat soga alami perpaduan kulit kayu tingi, tegeran, dan jambal simbol membumi.',
      philosophy: 'Warna cokelat tanah melambangkan kerendahan hati manusia yang berasal dari tanah dan kembali ke tanah.',
      history: 'Merupakan puncak teknologi pewarnaan alami masa Kesultanan Mataram yang bertahan ratusan tahun.',
      visualCharacteristics: 'Nuansa cokelat keemasan tua khas dengan garis kontur hitam pekat dan latar krem gading.',
      usageOccasion: 'Busana adat keraton harian, perjamuan resmi, dan prosesi adat sakral.',
      primaryColor: Color(0xFF5C3317),
      secondaryColor: Color(0xFFC68642),
      tag: 'Soga Alami',
    ),
    BatikHeritageItem(
      id: 'batik_tambal',
      rawLabels: ['batik-tambal', 'tambal'],
      name: 'Batik Tambal Sewu',
      region: 'Yogyakarta',
      province: 'D.I. Yogyakarta',
      island: 'Jawa',
      category: 'Filosofis Spiritual',
      shortDescription: 'Kumpulan ragam motif penambal kekurangan diri dan penolak bala penyembuh ragawi.',
      philosophy: 'Menambal berarti memperbaiki apa yang rusak, introspeksi diri, serta doa penyembuhan penyakit.',
      history: 'Dahulu diselimutkan kepada orang yang sedang sakit dengan keyakinan memancarkan energi kesembuhan.',
      visualCharacteristics: 'Bidang-bidang segitiga atau jajaran genjang berisi aneka motif batik mikro yang tersusun selang-seling.',
      usageOccasion: 'Upacara tolak bala, busana adat para tetua, dan koleksi spiritual.',
      primaryColor: Color(0xFF3E2723),
      secondaryColor: Color(0xFFBCAAA4),
      tag: 'Penyembuh Jiwa',
    ),
    BatikHeritageItem(
      id: 'dki_ondel_ondel',
      rawLabels: ['DKI_Ondel_Ondel', 'ondel_ondel', 'ondel-ondel'],
      name: 'Batik Betawi Ondel-Ondel',
      region: 'Jakarta Pusat & Timur',
      province: 'DKI Jakarta',
      island: 'Jawa',
      category: 'Kontemporer Kultural',
      shortDescription: 'Ikon sepasang boneka raksasa penjaga kota Jakarta pembawa keceriaan dan tolak bala.',
      philosophy: 'Ondel-ondel melambangkan pelindung keselamatan warga dan simbol kehangatan silaturahmi Betawi.',
      history: 'Diadaptasikan dari kesenian Barongan Betawi abad ke-16 menjadi motif wastra modern sejak era Ali Sadikin.',
      visualCharacteristics: 'Figur boneka Ondel-Ondel laki-laki dan perempuan dengan hiasan kembang kelapa warna cerah ceria.',
      usageOccasion: 'Peringatan HUT Kota Jakarta, festival budaya Betawi, dan seragam dinas provinsi.',
      primaryColor: Color(0xFFD32F2F),
      secondaryColor: Color(0xFFFFB300),
      tag: 'Ikon Ibukota',
    ),
    BatikHeritageItem(
      id: 'jawa_timur_pring',
      rawLabels: ['Jawa_Timur_Pring', 'pring_sedapur', 'pring'],
      name: 'Batik Pring Sedapur Magetan',
      region: 'Magetan',
      province: 'Jawa Timur',
      island: 'Jawa',
      category: 'Pedalaman Jawa Timur',
      shortDescription: 'Rumpun bambu nan rimbun simbol kerukunan, ketahanan hidup, dan kesetiaan persaudaraan.',
      philosophy: 'Bambu yang tumbuh berumpun (sedapur) mengajarkan persatuan gotong royong dan kelenturan jiwa.',
      history: 'Lahir dari kearifan desa Sidomukti di lereng Gunung Lawu, Magetan, terinspirasi hutan bambu asri.',
      visualCharacteristics: 'Batang-batang bambu tegak dengan rumpun daun meruncing anggun berhias burung merak.',
      usageOccasion: 'Busana khas Jawa Timuran, resepsi keluarga, dan festival seni budaya.',
      primaryColor: Color(0xFF1B4332),
      secondaryColor: Color(0xFF80B918),
      tag: 'Harmoni Bambu',
    ),
    BatikHeritageItem(
      id: 'lampung_gajah',
      rawLabels: ['Lampung_Gajah', 'gajah_lampung', 'lampung'],
      name: 'Batik Lampung Gajah & Siger',
      region: 'Bandar Lampung',
      province: 'Lampung',
      island: 'Sumatera',
      category: 'Sumatera Selatan',
      shortDescription: 'Pesona gajah Way Kambas dipadu mahkota Siger lambang kehormatan dan keagungan Sakai Sambayan.',
      philosophy: 'Gajah melambangkan kekuatan dan kebijaksanaan, sedangkan Siger melambangkan martabat luhur wanita Lampung.',
      history: 'Hasil kolaborasi perajin batik dengan motif tenun Tapis Lampung yang telah ada sejak abad ke-12.',
      visualCharacteristics: 'Figur gajah gagah dengan ornamen mahkota Siger bertatah pucuk rebung warna emas dan merah tua.',
      usageOccasion: 'Festival Krakatau, pernikahan adat Lampung Begawi, dan pakaian formal kelembagaan.',
      primaryColor: Color(0xFF8B0000),
      secondaryColor: Color(0xFFFFD700),
      tag: 'Kehormatan Siger',
    ),
    BatikHeritageItem(
      id: 'madura_mataketeran',
      rawLabels: ['Madura_Mataketeran', 'mataketeran', 'madura'],
      name: 'Batik Madura Mata Keteran',
      region: 'Pamekasan & Sumenep',
      province: 'Jawa Timur',
      island: 'Jawa',
      category: 'Pesisir Madura',
      shortDescription: 'Mata burung tekukur yang tajam dan awas mencerminkan keuletan serta kejujuran pesisir.',
      philosophy: 'Kewaspadaan menjaga kehormatan diri dan ketajaman intuisi dalam mengarungi ombak kehidupan.',
      history: 'Warisan seni batik keraton Sumenep yang menyebar ke sentra perajin rakyat di bumi Pamekasan.',
      visualCharacteristics: 'Bintik-bintik konsentris melingkar seperti mata burung dipadu sulur daun dan warna merah kuning berani.',
      usageOccasion: 'Busana pesta adat Madura, festival Karapan Sapi, dan upacara adat pesisir.',
      primaryColor: Color(0xFFB71C1C),
      secondaryColor: Color(0xFFFFA000),
      tag: 'Karakter Pesisir',
    ),
    BatikHeritageItem(
      id: 'ntb_lumbung',
      rawLabels: ['NTB_Lumbung', 'lumbung', 'lumbung_sasak'],
      name: 'Batik Sasambo Lumbung',
      region: 'Lombok & Sumbawa',
      province: 'Nusa Tenggara Barat',
      island: 'Nusa Tenggara',
      category: 'Sasambo Etnik',
      shortDescription: 'Lumbung padi tradisional Sasak (Pantek) lambang ketahanan pangan dan kemakmuran bumi.',
      philosophy: 'Lumbung melambangkan rasa syukur atas panen, kearifan menyimpan bekal masa depan, dan gotong royong.',
      history: 'Penyatuan tiga etnis utama NTB: Sasak, Samawa, dan Mbojo (Sasambo) dalam inovasi seni batik modern.',
      visualCharacteristics: 'Atap lumbung melengkung khas suku Sasak dipadu motif tenun ikat Sumbawa dan warna alam pulau.',
      usageOccasion: 'Festival Bau Nyale, perhelatan kebudayaan Sasambo, dan busana formal provinsi NTB.',
      primaryColor: Color(0xFF4A3B32),
      secondaryColor: Color(0xFFE28743),
      tag: 'Bumi Sasambo',
    ),
    BatikHeritageItem(
      id: 'papua_asmat',
      rawLabels: ['Papua_Asmat', 'asmat', 'batik-asmat'],
      name: 'Batik Ukir Asmat',
      region: 'Agats & Asmat',
      province: 'Papua Selatan',
      island: 'Papua',
      category: 'Etnik Pedalaman',
      shortDescription: 'Guratan mistis perisai kayu leluhur suku Asmat perajin ukir kayu terhebat di dunia.',
      philosophy: 'Penghormatan sakral kepada arwah leluhur, keberanian ksatria penjaga tanah ulayat, dan kearifan alam bakau.',
      history: 'Tradisi ukir perisai perang (Jemesi) yang diabadikan ke dalam media batik kain modern sejak tahun 1990.',
      visualCharacteristics: 'Pola spiral kait khas Asmat berulang dengan warna tanah terakota, hitam jelaga, dan putih kapur.',
      usageOccasion: 'Pesta Budaya Asmat, festival seni etnik Papua, dan busana budaya Nusantara.',
      primaryColor: Color(0xFF261C14),
      secondaryColor: Color(0xFFBA4A00),
      tag: 'Ukir Asmat',
    ),
    BatikHeritageItem(
      id: 'papua_tifa',
      rawLabels: ['Papua_Tifa', 'tifa', 'batik-tifa'],
      name: 'Batik Papua Tifa Harmoni',
      region: 'Merauke & Sentani',
      province: 'Papua & Papua Selatan',
      island: 'Papua',
      category: 'Alat Musik Etnik',
      shortDescription: 'Ketukan ritmis gendang tifa pengiring tarian persaudaraan dan pemersatu warga.',
      philosophy: 'Dentuman tifa memanggil kesadaran bersama, mempererat ikatan kekerabatan adat, dan mengobarkan semangat.',
      history: 'Tifa adalah instrumen purba sakral yang mengiringi setiap upacara adat di seantero tanah Papua.',
      visualCharacteristics: 'Siluet gendang kayu ramping dengan kulit biawak dipadu ornamen flora rimba dan sulur Papua.',
      usageOccasion: 'Tarian adat pergaulan, festival Danau Sentani, dan upacara penyambutan adat.',
      primaryColor: Color(0xFF4E342E),
      secondaryColor: Color(0xFFFF8F00),
      tag: 'Ritme Persaudaraan',
    ),
    BatikHeritageItem(
      id: 'sulawesi_selatan_lontara',
      rawLabels: ['Sulawesi_Selatan_Lontara', 'lontara', 'sulsel_lontara'],
      name: 'Batik Sutra Lontara',
      region: 'Makassar & Wajo',
      province: 'Sulawesi Selatan',
      island: 'Sulawesi',
      category: 'Aksara & Sutra Sengkang',
      shortDescription: 'Keanggunan aksara kuno Lontara Bugis-Makassar pada helaian kain sutra Sengkang nan masyhur.',
      philosophy: 'Aksara Lontara mengandung petuah luhur Siri Na Pacce: harga diri, kejujuran budi, dan solidaritas persaudaraan.',
      history: 'Dituliskan dari manuskrip kuno abad ke-14 I La Galigo, diaplikasikan ke kain sutra tenun perajin Danau Tempe.',
      visualCharacteristics: 'Karakter aksara bersudut tegas berpadu motif geometris Toraja Paqtedong di atas kain sutra cerah.',
      usageOccasion: 'Pesta adat perkawinan Bugis-Makassar, upacara pelantikan gelar adat, dan pakaian kehormatan.',
      primaryColor: Color(0xFF1F3A3D),
      secondaryColor: Color(0xFFE5A93B),
      tag: 'Siri Na Pacce',
    ),
    BatikHeritageItem(
      id: 'sumatera_utara_boraspati',
      rawLabels: ['Sumatera_Utara_Boraspati', 'boraspati', 'boraspati_ni_tano'],
      name: 'Batik Batak Boraspati',
      region: 'Toba & Tapanuli',
      province: 'Sumatera Utara',
      island: 'Sumatera',
      category: 'Gorga Batak Toba',
      shortDescription: 'Boraspati ni Tano cecak mitologis pelindung kesuburan bumi dan penegak ketenteraman rumah.',
      philosophy: 'Cecak dapat hidup di mana pun dan memakan hama, lambang adaptasi ksatria Batak di perantauan dan pelindung.',
      history: 'Diadaptasikan dari ukiran kayu Gorga dinding Rumah Bolon suku Batak Toba ke seni wastra batik.',
      visualCharacteristics: 'Figur cecak bercabang dua dipadu pola melingkar Gorga Simeol-meol berwarna merah, hitam, dan putih.',
      usageOccasion: 'Pesta adat Horja Batak, upacara pernikahan mangulosi, dan perhelatan kebudayaan Toba.',
      primaryColor: Color(0xFF8B0000),
      secondaryColor: Color(0xFFF5F5DC),
      tag: 'Gorga Batak',
    ),
    BatikHeritageItem(
      id: 'bali_barong_khusus',
      rawLabels: ['Bali_Barong', 'barong'],
      name: 'Batik Bali Barong Agung',
      region: 'Gianyar',
      province: 'Bali',
      island: 'Bali',
      category: 'Mitos & Spiritual',
      shortDescription: 'Wajah Barong Ket pelindung kebajikan dalam pertarungan abadi dharma melawan adharma.',
      philosophy: 'Kemenangan dharma (kebaikan) atas adharma (kejahatan) dan keseimbangan spiritual kosmos Tri Hita Karana.',
      history: 'Kesenian sakral Barong Gianyar yang dituangkan ke dalam kain batik lukis tangan oleh maestro perajin Ubud.',
      visualCharacteristics: 'Figur topeng Barong bertaring ramah bermahkota bunga kamboja berhias sulur ukiran Bali kencana.',
      usageOccasion: 'Upacara adat Pura, festival seni Pesta Kesenian Bali (PKB), dan busana pesta kriya.',
      primaryColor: Color(0xFF7B1113),
      secondaryColor: Color(0xFFD4AF37),
      tag: 'Dharma Pelindung',
    ),
    BatikHeritageItem(
      id: 'papua_cendrawasih_etnik',
      rawLabels: ['Papua_Cendrawasih'],
      name: 'Batik Cendrawasih Etnik',
      region: 'Jayapura',
      province: 'Papua',
      island: 'Papua',
      category: 'Satwa Langka Nusantara',
      shortDescription: 'Ikon burung surga khas tanah Papua dalam kemilau warna bulu keemasan nan sakral.',
      philosophy: 'Kecantikan alami yang harus dijaga kelestariannya serta rasa bangga akan kekayaan hayati rimba Papua.',
      history: 'Motif fauna endemik Papua yang menjadi duta persahabatan budaya Nusantara ke kancah internasional.',
      visualCharacteristics: 'Siluet bulu ekor mengembang indah dengan ornamen garis tifa melingkar di sekeliling sayap.',
      usageOccasion: 'Pekan Olahraga Nasional, resepsi duta budaya, dan festival wastra nasional.',
      primaryColor: Color(0xFF4A154B),
      secondaryColor: Color(0xFFF39C12),
      tag: 'Burung Surga',
    ),
  ];

  /// The featured Motif of the Day
  static const BatikHeritageItem motifOfTheDay = BatikHeritageItem(
    id: 'parang_rusak',
    rawLabels: ['batik-parang', 'batik-parang-rusak', 'parang'],
    name: 'Parang Rusak Barong',
    region: 'Yogyakarta & Surakarta',
    province: 'D.I. Yogyakarta',
    island: 'Jawa',
    category: 'Keraton Mataram',
    shortDescription: 'Simbol ksatria dalam mengendalikan hawa nafsu dan kesinambungan budi pekerti luhur.',
    philosophy:
          'Pola lereng diagonal yang membentuk jajaran huruf "S" tak terputus melambangkan kesinambungan garis keturunan, kepemimpinan bijaksana, dan keberanian melawan godaan batin.',
    history:
          'Diciptakan oleh Sultan Agung Hanyokrokusumo saat bermeditasi di pesisir Parangtritis. Merupakan salah satu pola agung tertua peradaban Mataram.',
    visualCharacteristics:
          'Jajaran motif parang berukuran besar (>8 cm) dengan aksen mlinjon tajam dan warna soga sogan cokelat tanah.',
    usageOccasion: 'Busana pusaka upacara keraton dan prosesi pernikahan agung.',
    primaryColor: Color(0xFF5A2A18),
    secondaryColor: Color(0xFFC8963E),
    tag: 'Batik of the Day',
  );

  /// Trending cultural motifs for the dashboard
  static List<BatikHeritageItem> get trendingMotifs => [
        allMotifs[1], // Megamendung
        allMotifs[2], // Kawung
        allMotifs[3], // Bali Barong
        allMotifs[5], // Pintu Aceh
        allMotifs[6], // Dayak
        allMotifs[7], // Papua Cendrawasih
      ];

  /// Resolves any model prediction raw label to a rich cultural heritage object.
  static BatikHeritageItem lookup(String rawLabel) {
    final cleaned = rawLabel.toLowerCase().replaceAll('-', '_');
    // 1. Exact match pass
    for (final item in allMotifs) {
      for (final l in item.rawLabels) {
        final cl = l.toLowerCase().replaceAll('-', '_');
        if (cleaned == cl) {
          return item;
        }
      }
    }
    // 2. Substring match pass
    for (final item in allMotifs) {
      for (final l in item.rawLabels) {
        final cl = l.toLowerCase().replaceAll('-', '_');
        if (cleaned.contains(cl) || cl.contains(cleaned)) {
          return item;
        }
      }
    }
    // Fallback: Generate a clean placeholder item
    final formattedName = rawLabel.replaceAll('-', ' ').replaceAll('_', ' ');
    return BatikHeritageItem(
      id: cleaned,
      rawLabels: [rawLabel],
      name: formattedName,
      region: 'Nusantara',
      province: 'Indonesia',
      island: 'Indonesia',
      category: 'Tradisional',
      shortDescription: 'Ragam hias warisan budaya wastra nusantara dengan nilai filosofis mendalam.',
      philosophy:
          'Setiap guratan canting dan malam pada motif wastra nusantara menyimpan doa, harapan kebajikan, dan perwujudan kearifan lokal leluhur bangsa.',
      history:
          'Batik telah diakui oleh UNESCO sebagai Warisan Kemanusiaan untuk Budaya Lisan dan Nonbendawi (Masterpiece of the Oral and Intangible Heritage of Humanity) sejak 2 Oktober 2009.',
      visualCharacteristics:
          'Karakteristik visual ornamen khas nusantara dengan harmoni komposisi garis, isen-isen tradisional, dan pewarnaan khas daerah.',
      usageOccasion: 'Acara kebudayaan, pertemuan formal, dan busana harian Nusantara.',
      primaryColor: const Color(0xFF8D4925),
      secondaryColor: const Color(0xFFC8963E),
      tag: 'Warisan Nusantara',
    );
  }
}
